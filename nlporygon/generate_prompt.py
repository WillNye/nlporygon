from pathlib import Path

import aiofiles
from jinja2 import Template

from nlporygon.models import Config, Table, SchemaAlias, Database


def int_to_str(n: int) -> str:
    """
    Convert integer to Excel-style column name.

    Examples:
        1 -> 'A'
        26 -> 'Z'
        27 -> 'AA'
        702 -> 'ZZ'
        703 -> 'AAA'
    """
    result = []
    while n > 0:
        n -= 1  # Adjust for 1-based indexing
        result.append(chr(ord('A') + (n % 26)))
        n //= 26

    return ''.join(reversed(result))


async def _generate_lookups(
    config: Config,
    tables: list[Table],
) -> SchemaAlias:
    table_alias_map = dict(to_alias=dict(), from_alias=dict())
    column_alias_map = dict(to_alias=dict(), from_alias=dict())
    data_type_alias_map = dict(to_alias=dict(), from_alias=dict())
    col_count = 0
    for t, table in enumerate(tables):
        t += 1
        t_alias = int_to_str(t)
        table_alias_map["to_alias"][table.name] = "t<" + t_alias + ">"
        table_alias_map["from_alias"][table.name] = t_alias
        for c, column in enumerate(table.columns):
            col_count += 1
            c_alias = int_to_str(col_count)
            if column.data_type not in data_type_alias_map["to_alias"]:
                data_type_alias_map["to_alias"][column.data_type] = "d<" + c_alias + ">"
                data_type_alias_map["from_alias"][column.data_type] = c_alias

            if column.name not in column_alias_map["to_alias"]:
                column_alias_map["to_alias"][column.name] = "c<" + c_alias + ">"
                column_alias_map["from_alias"][column.name] = c_alias

    schema_alias = SchemaAlias(
        table_alias_map=table_alias_map,
        column_alias_map=column_alias_map,
        data_type_alias_map=data_type_alias_map,
    )
    await schema_alias.write(config.output_path)

    return schema_alias


def _get_prompt_prefix(db: Database) -> str:
    prompt_path = Path(__file__).parent / "data" / "sys_prompt_header.j2"
    prompt = prompt_path.read_text()
    return Template(prompt).render(
        db_type=db.database_type.value,
        db_version=db.database_version
    )


async def generate_prompts(config: Config, db: Database):
    prompt = _get_prompt_prefix(db)
    tables = await Table.load_all(config.schema_path)
    schema_alias = await _generate_lookups(config, tables)
    table_names = {t.name for t in tables}

    for table in tables:
        prompt = f"{prompt}\n{_compress_schema(schema_alias, table, table_names)}"

    prompt_path = config.prompt_path / "sys_prompt.txt"
    async with aiofiles.open(prompt_path, mode="w") as f:
        await f.write(prompt)

    await create_legends(config, schema_alias)


def _compress_schema(
    schema_alias: SchemaAlias,
    table: Table,
    table_names: set[str],
) -> str:
    table_name = schema_alias.get_table_alias(table.name)
    schema_str = f"{table_name}("
    for column in table.columns:
        column_name = schema_alias.get_column_alias(column.name)
        data_type = schema_alias.get_data_type_alias(column.data_type)
        schema_str += f"{column_name} {data_type}"
        if column.sub_data_type:
            schema_str += f" --> {column.sub_data_type}"

        if relationships := [r for r in column.relationships if r.table in table_names]:
            schema_str += " REF["
            for relationship in relationships:
                ref_table = schema_alias.get_table_alias(relationship.table)
                ref_column = schema_alias.get_column_alias(relationship.column)
                schema_str += f"{ref_table}({ref_column})|"
            schema_str = schema_str[:-1] + "]"

        schema_str += ", "

    schema_str = schema_str[:-2]
    schema_str += ")"
    return schema_str


async def create_legends(config: Config, schema_alias: SchemaAlias):
    legend = "Here is a legend of the Tables in the DB. The format is $alias->$properName where $alias is the prompt provided earlier\n"
    for name, alias in schema_alias.table_alias_map["to_alias"].items():
        legend += f"{alias}->{name}\n"

    prompt_path = config.prompt_path / "table_legend.txt"
    async with aiofiles.open(prompt_path, mode="w") as f:
        await f.write(legend)

    legend = "Here is a legend of the Columns in the DB. The format is $alias->$properName where $alias is the prompt provided earlier\n"
    for name, alias in schema_alias.column_alias_map["to_alias"].items():
        legend += f"{alias}->{name}\n"

    prompt_path = config.prompt_path / "column_legend.txt"
    async with aiofiles.open(prompt_path, mode="w") as f:
        await f.write(legend)

    legend = "Here is a legend of the DataTypes in the DB. The format is $alias->$properName where $alias is the prompt provided earlier\n"
    for name, alias in schema_alias.data_type_alias_map["to_alias"].items():
        legend += f"{alias}->{name}\n"

    prompt_path = config.prompt_path / "data_type_legend.txt"
    async with aiofiles.open(prompt_path, mode="w") as f:
        await f.write(legend)
