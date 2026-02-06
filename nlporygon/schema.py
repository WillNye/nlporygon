"""
Core data models for nlporygon.

Defines configuration, database abstractions, table/column schemas, and the
alias mapping system used for prompt compression.
"""
import asyncio
import json
from pathlib import Path
from typing import Optional, Union

import aiofiles
import yaml
from pydantic import BaseModel, Field

from nlporygon.models import is_empty_val, YamlDump


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


