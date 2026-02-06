"""
LLM agent for converting natural language to SQL queries.

Uses Claude with compressed schema prompts. Supports automatic retry with error feedback
when queries fail, and "context queries" where the LLM can request intermediate data
(e.g., enum values, sample rows) before generating the final query.
"""
import dataclasses
import re
from pathlib import Path
from textwrap import dedent
from typing import Optional

import orjson
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from anthropic.types import TextBlockParam, CacheControlEphemeralParam, MessageParam

from nlporygon.logger import logger
from nlporygon.models import Config, Database, AgentConfig
from nlporygon.query_store import QueryStore

# Prefix for intermediate queries where LLM requests data (e.g., enum values) before final SQL
CONTEXT_FLAG = "INTERNAL_CONTEXT_QUERY"

# Pre-compiled forbidden SQL patterns for efficient validation
# Each tuple: (compiled_pattern, human_readable_name)
_FORBIDDEN_PATTERNS: list[tuple[re.Pattern, str]] = [
    # DDL statements
    (re.compile(r'\b(CREATE|ALTER|DROP)\s+(TABLE|INDEX|VIEW|DATABASE|SCHEMA|FUNCTION|PROCEDURE|TRIGGER)\b', re.IGNORECASE), 'DDL statement'),
    (re.compile(r'\bTRUNCATE\s+TABLE\b', re.IGNORECASE), 'TRUNCATE statement'),

    # DML statements (non-SELECT)
    (re.compile(r'\bINSERT\s+INTO\b', re.IGNORECASE), 'INSERT statement'),
    (re.compile(r'\bUPDATE\s+\w+\s+SET\b', re.IGNORECASE), 'UPDATE statement'),
    (re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE), 'DELETE statement'),
    (re.compile(r'\bMERGE\s+INTO\b', re.IGNORECASE), 'MERGE statement'),

    # Permission/access control
    (re.compile(r'\b(GRANT|REVOKE)\b', re.IGNORECASE), 'Permission statement'),

    # Transaction control
    (re.compile(r'\b(COMMIT|ROLLBACK)\b', re.IGNORECASE), 'Transaction control'),

    # Dangerous functions/commands
    (re.compile(r'\bEXEC(UTE)?\s*\(', re.IGNORECASE), 'EXECUTE statement'),
    (re.compile(r'\b(XP_|SP_|DBMS_|UTL_)\w+', re.IGNORECASE), 'Stored procedure'),
    (re.compile(r'\bLOAD\s+DATA\b', re.IGNORECASE), 'LOAD DATA statement'),
    (re.compile(r'\bINTO\s+(OUTFILE|DUMPFILE)\b', re.IGNORECASE), 'File output clause'),

    # SQL injection patterns
    (re.compile(r';\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)', re.IGNORECASE), 'Multiple statements'),
    (re.compile(r'--\s*$', re.IGNORECASE), 'SQL comment injection'),
    (re.compile(r'/\*.*\*/', re.IGNORECASE), 'Block comment injection'),
    (re.compile(r'\bUNION\s+ALL\s+SELECT\s+NULL', re.IGNORECASE), 'UNION NULL injection'),
    (re.compile(r"'\s*OR\s+'?1'?\s*=\s*'?1|OR\s+'1'\s*=\s*'1'", re.IGNORECASE), 'OR 1=1 injection'),
    (re.compile(r'\bSLEEP\s*\(\d+\)', re.IGNORECASE), 'SLEEP timing attack'),
    (re.compile(r'\bBENCHMARK\s*\(', re.IGNORECASE), 'BENCHMARK timing attack'),
    (re.compile(r'\bWAITFOR\s+DELAY\b', re.IGNORECASE), 'WAITFOR timing attack'),

    # File system access
    (re.compile(r'\b(READ_FILE|WRITE_FILE|LOAD_FILE)\s*\(', re.IGNORECASE), 'File access function'),
]


class UnsafeSQLError(Exception):
    """Raised when LLM generates SQL that contains forbidden/dangerous patterns."""

    def __init__(self, message: str, pattern_name: str, sql: str):
        self.pattern_name = pattern_name
        self.sql = sql
        super().__init__(message)


def _validate_sql_safety(sql: str) -> None:
    """
    Check SQL for dangerous patterns and raise UnsafeSQLError if found.

    Uses pre-compiled regex patterns for efficiency. This is a defense-in-depth
    measure - the LLM should only generate SELECT statements, but this catches
    any attempts at DDL, DML, or injection.
    """
    for pattern, name in _FORBIDDEN_PATTERNS:
        if pattern.search(sql):
            raise UnsafeSQLError(
                f"Unsafe SQL detected: {name}",
                pattern_name=name,
                sql=sql
            )


def _sanitize_llm_response(response) -> str:
    """Extracts SQL from LLM response, stripping markdown artifacts and outer LIMIT/OFFSET."""
    llm_response = response.content[0].text
    if CONTEXT_FLAG in llm_response:
        llm_response = re.split(CONTEXT_FLAG, llm_response, flags=re.IGNORECASE)[-1]
        llm_response = f"{CONTEXT_FLAG}\n{llm_response}"
    else:
        llm_response = re.split(r'SELECT', llm_response, flags=re.IGNORECASE)[-1]
        llm_response = f"SELECT{llm_response}"

    llm_response = llm_response.rstrip('`')

    # Remove only outer LIMIT and OFFSET (at end of query, preserves subqueries)
    llm_response = re.sub(r'\s+LIMIT\s+\d+\s+OFFSET\s+\d+\s*$', '', llm_response, flags=re.IGNORECASE)
    llm_response = re.sub(r'\s+LIMIT\s+\d+\s*$', '', llm_response, flags=re.IGNORECASE)
    llm_response = re.sub(r'\s+OFFSET\s+\d+\s*$', '', llm_response, flags=re.IGNORECASE)

    llm_response = llm_response.strip()

    # Validate SQL is safe before returning
    _validate_sql_safety(llm_response)

    return llm_response


def _set_query_limit_offset(
    query: str,
    limit: int,
    offset: Optional[int] = 0,
) -> str:
    return f"{query}\nLIMIT\n{limit}\nOFFSET {offset}"


class AgentResponse(BaseModel):
    user_prompt: str
    llm_response: str
    db_data: list[dict]


@dataclasses.dataclass
class DbAgent:
    """
    Converts natural language to SQL using Claude with compressed schema context.

    Handles automatic retry with error feedback when queries fail, and supports
    context queries where the LLM can request intermediate data before final SQL.
    """
    agent: AsyncAnthropic
    config: Config
    db: Database
    prompt_path: Optional[Path] = None
    _system_prompt: Optional[list[TextBlockParam]] = None

    def __post_init__(self):
        if self.prompt_path is None:
            self.prompt_path = self.config.prompt_path

    @property
    def agent_config(self) -> AgentConfig:
        return self.config.agent_config

    @property
    def system_prompt(self) -> list[TextBlockParam]:
        """
        Loads and caches the system prompt from sys_prompt.txt + legend files.
        Applies prompt caching to the last block for cost efficiency on repeated calls.
        """
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
            model=self.agent_config.model_version,
            system=self.system_prompt,  # Cached after first call
            messages=message_history,
            max_tokens=self.agent_config.max_tokens,
            temperature=0,  # Deterministic output for consistent SQL generation
            timeout=self.agent_config.timeout,
        )
        try:
            llm_response = _sanitize_llm_response(response)
            message_history.append(MessageParam(role="assistant", content=llm_response))
        except UnsafeSQLError:
            raise

        return llm_response

    async def query(
        self,
        message: str,
        limit: int,
        offset: int
    ) -> AgentResponse:
        """
        Converts natural language to SQL and executes it.

        Handles up to max_query_attempts retries with error feedback, and up to 3 context
        queries per attempt where the LLM can request data before generating final SQL.
        Returns empty list if all attempts fail or return no data.
        """
        logger.info("Processing natural language query", message=message[:100])
        current_attempt = 0
        message_history = []
        message_prefix = ""
        response = AgentResponse(user_prompt=message, llm_response="", db_data=[])

        while current_attempt < self.agent_config.max_query_attempts:
            context_queries = 0
            current_attempt += 1
            logger.debug("Sending query to LLM", attempt=current_attempt)

            try:
                q_response = await self._send_message(
                    f"{message_prefix}{message}",
                    message_history
                )
            except UnsafeSQLError as e:
                message_prefix = f"The query you wrote is unsafe: {e}. "
                continue

            while q_response.startswith(CONTEXT_FLAG) and context_queries < self.agent_config.max_context_queries:
                context_queries += 1
                logger.debug("Processing context query", context_query_num=context_queries)
                try:
                    db_data = await self.db.execute(
                        _set_query_limit_offset(
                            q_response.replace(CONTEXT_FLAG, ""),
                            1000
                        )
                    )
                    if db_data:
                        msg = orjson.dumps(db_data).decode()
                    else:
                        msg = "The query didn't return any data."

                    q_response = await self._send_message(
                        msg,
                        message_history
                    )
                except UnsafeSQLError as e:
                    message_prefix = f"The query you wrote is unsafe: {e}. "
                    q_response = None
                    break
                except Exception as e:
                    logger.warning("Context query failed", error=str(e))
                    q_response = await self._send_message(
                        f"The context query failed with the error: {e}",
                        message_history
                    )
                    break

            if not q_response:
                continue

            try:
                logger.debug("Executing generated SQL", query=q_response)
                db_data = await self.db.execute(
                    _set_query_limit_offset(q_response, limit, offset)
                )
                if db_data:
                    logger.info("Query successful", row_count=len(db_data))
                    response.llm_response = q_response
                    response.db_data = db_data
                    return response
                logger.debug("Query returned no data, retrying")
                message_prefix = "The query didn't return any data. Look closely at the query and try again. "
            except Exception as e:
                logger.warning("Query execution failed", attempt=current_attempt, error=str(e))
                message_prefix = f"The query failed with the error: {e}. "

        logger.warning("All query attempts exhausted", max_attempts=self.agent_config.max_query_attempts)
        return response


@dataclasses.dataclass
class MainAgent(DbAgent):
    """
    Extends DbAgent to support partitioned schemas and query caching.

    When multiple partitions exist, uses an LLM call to route incoming queries
    to the appropriate partition's agent based on the query content.

    If a query_store is provided, checks for semantically similar saved queries
    before invoking the LLM.
    """
    query_store: Optional[QueryStore] = None
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
            )
            self._partition_agent_map[partition.name] = db_agent

        return self._partition_agent_map

    @property
    def partition_name_map(self) -> dict:
        return self._partition_name_map

    async def query(self, message: str, limit: int, offset: int) -> AgentResponse:
        # Check saved queries first (if query_store is configured)
        if self.query_store:
            cached = await self.query_store.lookup(message)
            if cached:
                logger.info("Using cached query", similarity_match=cached.user_message[:50])
                try:
                    db_data = await self.db.execute(_set_query_limit_offset(cached.query, limit, offset))
                    return AgentResponse(
                        user_prompt=message,
                        llm_response=cached.query,
                        db_data=db_data
                    )
                except Exception as e:
                    logger.warning("Cached query execution failed, falling back to LLM", error=str(e))

        # Normal flow: route to partition and generate query
        partition_agent_map = self.partition_agent_map
        if len(partition_agent_map) == 1:
            agent = list(partition_agent_map.values())[0]
            logger.debug("Using single partition agent")
        else:
            partition_name = await self._select_partition(message)
            logger.info("Routed query to partition", partition=partition_name)
            agent = partition_agent_map[partition_name]

        response = await agent.query(message, limit, offset)
        if self.query_store and response.db_data:
            await self.query_store.save(
                message,
                response.llm_response,
                False
            )

        return response

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
            model=self.agent_config.model_version,
            system=system_prompt,
            messages=[MessageParam(role="user", content=message)],
            max_tokens=200,
            temperature=0,  # Deterministic routing
            timeout=self.agent_config.timeout,
        )
        selected_partition = response.content[0].text.strip()

        # Find best matching description (exact match or closest)
        if selected_partition in self.partition_name_map:
            return selected_partition

        # Fallback: find description that contains the response or vice versa
        for name in self.partition_name_map.keys():
            if selected_partition.lower() in name.lower() or name.lower() in selected_partition.lower():
                return name

        raise ValueError(f"{selected_partition} is not a valid partition name")
