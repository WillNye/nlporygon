"""
Core data models for nlporygon.

Defines configuration, database abstractions, table/column schemas, and the
alias mapping system used for prompt compression.
"""
import asyncio
import json
import re
from pathlib import Path
from typing import Callable, Optional, Any, Union

import aiofiles
import yaml
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

def is_empty_val(value):
    return not value


class Database(BaseModel):
    """
    Database connection wrapper using SQLAlchemy async engine.
    Supports any SQLAlchemy-compatible backend (PostgreSQL, SQLite, DuckDB, Snowflake, etc.)
    """
    name: str
    database_type: str
    database_version: str | None = None
    connection: Union[AsyncEngine, Engine]
    timeout: Optional[int] = 30

    model_config = {"arbitrary_types_allowed": True}

    async def execute(
        self,
        query: str,
        query_params: Optional[dict] = None,
        **kwargs
    ) -> list[dict]:
        """
        Executes a SQL query and returns results as a list of dictionaries.

        Args:
            query: SQL query string. Use :param_name for parameter binding.
            query_params: Optional dict of parameter values.

        Returns:
            List of dicts, one per row, with column names as keys.
        """
        if isinstance(self.connection, Engine):
            with self.connection.connect() as conn:
                result = conn.execute(
                    text(query),
                    query_params or {}
                )
                rows = result.fetchall()
        else:
            async with self.connection.connect() as conn:
                result = await conn.execute(
                    text(query),
                    query_params or {}
                )
                rows = result.fetchall()

        if not result.keys():
            return []

        column_names = list(result.keys())
        return [dict(zip(column_names, row)) for row in rows]


class ColumnConfig(BaseModel):
    # Global rules to ignore relationships for columns specified in this list
    global_ignore_column_rules: Optional[list[str]] = Field(default_factory=list, exclude_if=is_empty_val)

    # Number of records to use when checking for a relationship between columns
    sample_size: Optional[int] = 20_000

    # Number of records to use when creating bloom filter for relationship check
    bloom_size: Optional[int] = 1_000_000

    # Bloom filter error rate
    error_rate: Optional[float] = 0.01


class BaseTableRule(BaseModel):
    """
    Regex-based table filtering. Ignore rules take precedence over include rules.
    If no include rules are specified, all tables are included by default.
    """
    # Explicit rules where only tables that match at least 1 provided regex
    #   will be used for generating prompts
    include_table_rules: Optional[list[str]] = Field(default_factory=list, exclude_if=is_empty_val)

    # Rules to ignore tables matching any of the defined regexes when generating prompts.
    # This also means tables that were a match on include_table_rules are ignored
    #   if they match one of these.
    ignore_table_rules: Optional[list[str]] = Field(default_factory=list, exclude_if=is_empty_val)

    def get_matching_tables(self, tables: list["Table"]):
        def _is_match(_t: Table) -> bool:
            name = _t.name.replace('"', '')
            if any(
                re.match(r, name, flags=re.IGNORECASE)
                for r in self.ignore_table_rules
            ):
                return False

            if not self.include_table_rules:
                # Default behavior is to include all
                return True

            return any(
                re.match(r, name, flags=re.IGNORECASE)
                for r in self.include_table_rules
            )

        return [t for t in tables if _is_match(t)]


class TablePartitionConfig(BaseTableRule):
    name: str
    description: str
    tables: Optional[list["Table"]] = Field(default_factory=list, exclude_if=is_empty_val)


class CommonTableRule(BaseTableRule):
    pass


class TableConfig(BaseTableRule):
    common_table: Optional[CommonTableRule] = Field(default=None)
    partitions: Optional[list[TablePartitionConfig]] = Field(default_factory=list, exclude_if=is_empty_val)


class AgentConfig(BaseModel):
    model_version: Optional[str] = "claude-sonnet-4-5-20250929"
    max_tokens: Optional[int] = 2500
    max_query_attempts: Optional[int] = 2
    max_context_queries: Optional[int] = 3
    timeout: Optional[int] = 60


class Config(BaseModel):
    output_path: Union[Path, str]

    agent_config: Optional[AgentConfig] = Field(default_factory=AgentConfig, exclude_if=is_empty_val)
    column_relationships: Optional[ColumnConfig] = Field(default_factory=ColumnConfig, exclude_if=is_empty_val)
    table_config: Optional[TableConfig] = Field(default_factory=TableConfig, exclude_if=is_empty_val)

    @property
    def schema_path(self) -> Path:
        path = self.output_path / "schema"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def prompt_path(self) -> Path:
        path = self.output_path / "prompt"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_post_init(self, context: Any, /) -> None:
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        return super().model_post_init(context)


class ColumnRelationship(BaseModel):
    table: str
    column: str


class TableColumn(BaseModel):
    name: str
    data_type: str

    relationships: Optional[list[ColumnRelationship]] = Field(default_factory=list, exclude_if=is_empty_val)
    # Add context for columns where true value has been cast to another type
    # Example: "[1, 2, 3]"
    #   data_type would be VARCHAR
    #   sub_data_type would be INT[]
    # Now, the model will know to cast the column to INT[] before running a query
    sub_data_type: Optional[str] = Field(default=None)
    # Add visibility for keys on JSON columns
    nested_columns: Optional[list["TableColumn"]] = Field(default_factory=list, exclude_if=is_empty_val)

    @property
    def query_name(self) -> str:
        return f'"{self.name}"'


class Table(BaseModel):
    name: str
    columns: list[TableColumn]

    default_order: Optional[list[str]] = Field(default_factory=list, exclude_if=is_empty_val)
    ignore_columns: Optional[list[str]] = Field(default_factory=list, exclude_if=is_empty_val)

    async def write(self, path: Path):
        data = yaml.dump(
            self.model_dump(exclude_none=True),
            sort_keys=False,
            Dumper=YamlDump,
            default_flow_style=False
        )
        async with aiofiles.open(path.joinpath(self.file_name), "w") as f:
            await f.write(data)

    @property
    def file_name(self) -> str:
        # Convert db_name.main."my_table"
        # To my_table.yaml
        name = self.name.replace(".", "_").replace('"', "").lower()
        return f"{name}.yaml"

    @classmethod
    async def load(cls, path: Path) -> "Table":
        async with aiofiles.open(path) as f:
            content = await f.read()
            return cls.model_validate(
                yaml.safe_load(content)
            )

    @classmethod
    async def load_all(cls, path: Path) -> list["Table"]:
        path.mkdir(parents=True, exist_ok=True)

        return await asyncio.gather(
            *[
                cls.load(p) for p in path.iterdir()
                if p.is_file() and (p.name.endswith(".yaml") or p.name.endswith(".yml"))
            ]
        )


class SchemaAlias(BaseModel):
    """
    Bidirectional mappings between real names and compressed aliases.
    Used to compress prompts (real→alias) and decode LLM output (alias→real).
    """
    table_alias_map: dict[str, dict[str, str]]
    column_alias_map: dict[str, dict[str, str]]
    data_type_alias_map: dict[str, dict[str, str]]

    def get_table_alias(self, table_name: str) -> str:
        return self.table_alias_map["to_alias"][table_name]

    def get_table_name(self, table_alias: str) -> str:
        return self.table_alias_map["from_alias"][table_alias]

    def get_column_alias(self, column_name: str) -> str:
        return self.column_alias_map["to_alias"][column_name]

    def get_column_name(self, column_alias: str) -> str:
        return self.column_alias_map["from_alias"][column_alias]

    def get_data_type_alias(self, data_type: str) -> Union[str, None]:
        return self.data_type_alias_map["to_alias"][data_type]

    def get_enum_type(self, data_type: str) -> str:
        return self.data_type_alias_map["from_alias"][data_type]

    @property
    def file_name(self) -> str:
        return "schema_alias_map.json"

    async def write(self, path: Path):
        data = self.model_dump(exclude_none=True, mode='json')
        async with aiofiles.open(path.joinpath(self.file_name), "w") as f:
            await f.write(json.dumps(data, indent=4))

    @classmethod
    async def load(cls, path: Path) -> "SchemaAlias":
        async with aiofiles.open(path) as f:
            content = await f.read()
            return cls.model_validate(json.loads(content))


class YamlDump(yaml.SafeDumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(YamlDump, self).increase_indent(flow, False)
