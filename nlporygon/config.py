import re
from pathlib import Path
from typing import Optional, Union, Any

import aiofiles
import yaml

from pydantic import BaseModel, Field

from nlporygon.models import is_empty_val, YamlDump
from nlporygon.schema import Table

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

    _prompt_config: Optional["PromptConfig"] = None

    @property
    def prompt_config(self) -> "PromptConfig":
        if self._prompt_config is None:
            self._prompt_config = PromptConfig.load(self.prompt_path)

        return self._prompt_config

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


class PromptConfig(BaseModel):
    prompt_version: str

    async def write(self, path: Path):
        file_name = "prompt_config.yaml"
        data = yaml.dump(
            self.model_dump(exclude_none=True),
            sort_keys=False,
            Dumper=YamlDump,
            default_flow_style=False
        )
        async with aiofiles.open(path.joinpath(file_name), "w") as f:
            await f.write(data)

    @classmethod
    def load(cls, path: Path) -> "PromptConfig":
        file_name = "prompt_config.yaml"
        with open(path / file_name) as f:
            return cls.model_validate(
                yaml.safe_load(f.read())
            )
