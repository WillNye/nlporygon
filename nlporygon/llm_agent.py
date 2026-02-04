import dataclasses
import re
from pathlib import Path
from textwrap import dedent
from typing import Optional

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
    prompt_path: Optional[Path] = None
    max_query_attempts: Optional[int] = 2
    _system_prompt: Optional[list[TextBlockParam]] = None

    def __post_init__(self):
        if self.prompt_path is None:
            self.prompt_path = self.config.prompt_path

    @property
    def system_prompt(self) -> list[TextBlockParam]:
        if self._system_prompt:
            return self._system_prompt

        # Load files in specific order: sys_prompt first, then legends
        # This ensures the LLM sees instructions before the alias mappings
        prompt_files = []
        sys_prompt_file = self.prompt_path / "sys_prompt.txt"
        if sys_prompt_file.exists():
            prompt_files.append(sys_prompt_file)

        # Add legend files in consistent order
        for legend_name in ["table_legend.txt", "column_legend.txt", "data_type_legend.txt"]:
            legend_file = self.prompt_path / legend_name
            if legend_file.exists():
                prompt_files.append(legend_file)

        prompts: list[TextBlockParam] = []
        for path in prompt_files:
            prompts.append(TextBlockParam(type="text", text=path.read_text()))

        if prompts:
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

            while q_response.startswith(CONTEXT_FLAG) and context_queries < 3:
                context_queries += 1
                try:
                    db_data = await self.db.execute(q_response.replace(CONTEXT_FLAG, ""))
                    if db_data:
                        msg = orjson.dumps(db_data).decode()
                    else:
                        msg = "The query didn't return any data."

                    q_response = await self._send_message(
                        msg,
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


@dataclasses.dataclass
class MainAgent(DbAgent):
    _partition_agent_map: Optional[dict] = None
    _partition_name_map: Optional[dict] = None

    @property
    def partition_agent_map(self) -> dict:
        if self._partition_agent_map:
            return self._partition_agent_map

        self._partition_agent_map = {}
        self._partition_name_map = {}

        for partition in self.config.table_config.partitions:
            self._partition_name_map[partition.name] = partition.description

            prompt_path = self.prompt_path / partition.name
            db_agent = DbAgent(
                AsyncAnthropic(),
                self.config,
                self.db,
                prompt_path,
                self.max_query_attempts,
            )
            self._partition_agent_map[partition.name] = db_agent

        return self._partition_agent_map

    @property
    def partition_name_map(self) -> dict:
        return self._partition_name_map

    async def query(self, message: str) -> list[dict]:
        partition_agent_map = self.partition_agent_map
        if len(partition_agent_map) == 1:
            agent = list(partition_agent_map.values())[0]
        else:
            partition_name = await self._select_partition(message)
            agent = partition_agent_map[partition_name]

        return await agent.query(message)

    async def _select_partition(self, message: str) -> str:
        """Use LLM to select the best partition for the given query."""
        system_prompt = dedent("""You are a database query router. 
        Given a user's natural language query, determine which database partition is most relevant.
        
        Partition Format
        - Name : Description
        
        Available partitions:
        """)
        for desc, name in self.partition_name_map.items():
            system_prompt += f"- {name} : {desc}\n"

        system_prompt += dedent("""Respond with ONLY the exact partition name that best matches the query. 
        No explanation, just the partition name.""")

        response = await self.agent.messages.create(
            model=MODEL_VERSION,
            system=system_prompt,
            messages=[MessageParam(role="user", content=message)],
            max_tokens=200
        )
        selected_partition = response.content[0].text.strip()

        # Find best matching description (exact match or closest)
        if selected_partition in self.partition_name_map:
            return selected_partition

        # Fallback: find description that contains the response or vice versa
        for name in self.partition_name_map.keys():
            if selected_partition.lower() in name.lower() or name.lower() in selected_partition.lower():
                return name

        # Last resort: return first partition
        return list(self._partition_name_map.values())[0]
