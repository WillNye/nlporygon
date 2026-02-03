import dataclasses
import re
from typing import Optional, Tuple

import orjson
from anthropic import AsyncAnthropic
from anthropic.types import TextBlockParam, CacheControlEphemeralParam, MessageParam

from nlporygon.models import Config, Database

MODEL_VERSION = "claude-sonnet-4-5-20250929"
CONTEXT_FLAG = "INTERNAL_CONTEXT_QUERY"


def _sanitize_llm_response(response) -> str:
    llm_response = response.content[0].text
    if CONTEXT_FLAG in llm_response:
        llm_response = re.split(CONTEXT_FLAG, llm_response, flags=re.IGNORECASE)[-1]
        llm_response = f"{CONTEXT_FLAG}\n{llm_response}"
    else:
        llm_response = re.split(r'SELECT', llm_response, flags=re.IGNORECASE)[-1]
        llm_response = f"SELECT{llm_response}"

    return llm_response.rstrip('`')


@dataclasses.dataclass
class DbAgent:
    agent: AsyncAnthropic
    config: Config
    db: Database
    max_query_attempts: Optional[int] = 3
    _system_prompt: Optional[list[TextBlockParam]] = None

    @property
    def system_prompt(self) -> list[TextBlockParam]:
        if self._system_prompt:
            return self._system_prompt

        prompts: list[TextBlockParam] = []
        for path in self.config.prompt_path.iterdir():
            prompts.append(TextBlockParam(type="text", text=path.read_text()))

        prompts[-1]["cache_control"] = CacheControlEphemeralParam(type="ephemeral", ttl='1h')
        self._system_prompt = prompts
        return self._system_prompt

    async def _send_message(
        self,
        user_message: str,
        message_history: list[MessageParam]
    ) -> str:
        message_history.append(MessageParam(role="user", content=user_message))
        response = await self.agent.messages.create(
            model=MODEL_VERSION,
            system=self.system_prompt,  # Cached after first call
            messages=message_history,
            max_tokens=5000
        )
        llm_response = _sanitize_llm_response(response)

        message_history.append(MessageParam(role="assistant", content=llm_response))
        return llm_response

    async def query(self, message: str) -> list[dict]:
        current_attempt = 0
        message_history = []
        message_prefix = ""

        while current_attempt < self.max_query_attempts:
            context_queries = 0
            current_attempt += 1
            q_response = await self._send_message(
                f"{message_prefix}{message}",
                message_history
            )

            while q_response.startswith(CONTEXT_FLAG) and context_queries < 5:
                context_queries += 1
                try:
                    q_response = q_response.replace(CONTEXT_FLAG, "")
                    db_data = await self.db.execute(q_response)
                    q_response = await self._send_message(
                        orjson.dumps(db_data).decode(),
                        message_history
                    )
                except Exception as e:
                    q_response = await self._send_message(
                        f"The context query failed with the error: {e}",
                        message_history
                    )
                    break

            try:
                db_data = await self.db.execute(q_response)
                if db_data:
                    return db_data
                message_prefix = "The query didn't return any data. Look closely at the query and try again. "
            except Exception as e:
                message_prefix = f"The query failed with the error: {e}. "

        return []


