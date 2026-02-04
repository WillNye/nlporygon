"""
Builds Table/TableColumn schema definitions by introspecting the database.

Detects sub_data_type for columns storing serialized data (e.g., JSON strings that
should be cast to arrays). Sets default ordering for tables based on date columns,
primary keys, or ordinal position.
"""
import asyncio
import datetime
import decimal
import json
import uuid
from textwrap import dedent
from typing import Union, Any

from nlporygon.column_relationship import set_column_relationship
from nlporygon.logger import logger
from nlporygon.models import Database, Table, TableColumn, Config
from nlporygon.utils import get_query, is_date_or_dt_column, get_table_name

PYTHON_TO_SQL_TYPE_MAP = {
    # Python built-ins
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "BOOLEAN",
    bytes: "BLOB",
    bytearray: "BLOB",
    memoryview: "BLOB",

    # datetime module
    datetime.datetime: "TIMESTAMP",
    datetime.date: "DATE",
    datetime.time: "TIME",
    datetime.timedelta: "INTERVAL",

    # decimal
    decimal.Decimal: "NUMERIC",

    # uuid
    uuid.UUID: "UUID",  # or VARCHAR(36) for compatibility
}


async def generate_table_definitions(
    config: Config,
    db: Database,
) -> list[Table]:
    """
    Main entry point. Builds schema definitions for all accessible tables.

    Loads existing schemas (removing stale tables no longer in DB), creates/updates
    definitions with columns, relationships, and default ordering, then writes to YAML.
    """
    schema_dir = config.schema_path
    table_data = await _get_tables(db)
    table_names = {get_table_name(r) for r in table_data}

    existing_tables = await Table.load_all(schema_dir)
    table_definition_map: dict[str, Table] = {}
    for table in existing_tables:
        if table.name in table_names:
            table_definition_map[table.name] = table
        else:
            logger.warning("Removing table no longer found in database", table=table.name)
            schema_path = schema_dir / table.file_name
            schema_path.unlink(missing_ok=True)

    await upsert_basic_table_definition(
        db,
        table_data,
        table_definition_map,
    )
    await set_column_relationship(
        config,
        db,
        list(table_definition_map.values())
    )
    await upsert_table_ordering(
        db,
        table_data,
        table_definition_map,
    )

    logger.info("Writing schema definitions")
    tables = list(table_definition_map.values())
    await asyncio.gather(*[
        table.write(schema_dir)
        for table in tables
    ])

    logger.info("Table definition generation complete")
    return tables


async def _get_tables(db: Database) -> list[dict]:
    """Fetches table metadata, filtering out tables that can't be queried (permissions, etc.)."""
    get_tables = get_query("get_tables", db.database_type)
    response = []
    missing_tables = set()
    for table in await db.execute(get_tables):
        table_name = get_table_name(table)
        if table_name in missing_tables:
            continue

        try:
            await db.execute(f"SELECT * FROM {table_name} LIMIT 100")
        except Exception as e:
            logger.warning(f"Failed to get table", table=table_name, error=e)
            missing_tables.add(table_name)
            continue
        else:
            response.append(table)

    return response


async def upsert_basic_table_definition(
    db: Database,
    table_data: list[dict],
    table_definition_map: dict[str, Table],
):
    """
    Creates or updates Table objects with columns from DB metadata.
    Detects sub_data_type for columns that store serialized data.
    """
    logger.info("Creating base definition for new tables")
    for column_dict in table_data:
        table_name = get_table_name(column_dict)
        column_name = column_dict['column_name']

        table: Table = table_definition_map.get(
            table_name,
            Table(name=table_name, columns=[])
        )
        if table.name not in table_definition_map:
            table_definition_map[table_name] = table

        if columns := [f for f in table.columns if f.name == column_name]:
            column = columns[0]
        else:
            column = TableColumn(
                name=column_name,
                data_type=column_dict['data_type'],
            )
            table.columns.append(column)

        await _set_sub_data_type(db, table, column)


async def upsert_table_ordering(
    db: Database,
    table_data: list[dict],
    table_definition_map: dict[str, Table],
):
    """
    Sets default_order for tables. Priority: date/datetime column (best for time-series),
    then primary key columns, then first column as fallback.
    """
    logger.info("Setting default query order for all tables")
    pk_results = await db.execute(
        get_query("get_pks.sql", db.database_type)
    )
    pk_columns = [
        f"{get_table_name(r)} - {r['column_name']}"
        for r in pk_results
    ]
    for table_name, table in table_definition_map.items():
        if table.default_order:
            continue

        ordinal_date = None
        primary_keys = []
        ordinal_init = None

        for column_dict in table_data:
            if get_table_name(column_dict) != table_name:
                continue

            column_name = column_dict['column_name']
            data_type = column_dict['data_type']

            if not ordinal_date and is_date_or_dt_column(data_type):
                ordinal_date = [column_name]
                break  # ordinal_date takes precedence so skip other checks
            elif f"{table_name} - {column_name}" in pk_columns:
                primary_keys.append(column_name)
            elif column_dict['ordinal_position'] == 1:
                ordinal_init = [column_name]

        if ordinal_date:
            table.default_order = ordinal_date
        elif primary_keys:
            table.default_order = primary_keys
        else:
            table.default_order = ordinal_init

    logger.info("Setting default query order complete")


async def _set_sub_data_type(
    db: Database,
    table: Table,
    column: TableColumn,
):
    """
    Samples column data to detect if the stored type differs from the declared type.

    For example, a VARCHAR column storing "[1,2,3]" gets sub_data_type="INTEGER[]" so
    the LLM knows to cast it. For JSON columns, recursively builds nested_columns to
    expose the internal structure. Clears sub_data_type if values are inconsistent.
    """
    def _upsert_nested_columns(_val: Union[dict, list], key: str):
        if isinstance(_val, list):
            if len(_val) > 0 and isinstance(_val[0], dict):
                _val = _val[0]
            else:
                return

        for k, v in _val.items():
            if v is None:
                continue

            k = f"{key}->{k}"
            nested_col = None
            if nested_cols := [x for x in column.nested_columns if x.name == k]:
                nested_col = nested_cols[0]
                if (
                    not nested_col.data_type.startswith("JSON")
                    and not nested_col.data_type.endswith("[]")
                ):
                    continue

            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except json.decoder.JSONDecodeError:
                    pass

            if isinstance(v, dict) or isinstance(v, list):
                _dt = _get_data_type(v)
                if not _dt:
                    continue
                elif _dt.startswith("JSON"):
                    _upsert_nested_columns(v, k)
            else:
                _dt = PYTHON_TO_SQL_TYPE_MAP[type(v)]

            if nested_col and nested_col.data_type != _dt:
                # The default cast because it's the safest
                nested_col.data_type = "VARCHAR"
            elif not nested_col:
                column.nested_columns.append(
                    TableColumn(
                        name=k,
                        data_type=_dt,
                    )
                )

    sample_data = await db.execute(dedent(f"""
    SELECT DISTINCT {column.query_name} 
    FROM {table.name} 
    WHERE {column.query_name} IS NOT NULL LIMIT 10000
    """))
    for row in sample_data:
        val = row[column.name]
        if isinstance(val, str) and val.replace(" ", "") == "":
            # Treat empty string like null
            continue

        try:
            val = json.loads(val)
        except (json.decoder.JSONDecodeError, TypeError):
            if not isinstance(val, list) and not isinstance(val, dict):
                # Consistent casting can't be done for the column
                column.sub_data_type = None
                column.nested_columns = None
                return

        sub_data_type = _get_data_type(val)
        if not column.sub_data_type:
            column.sub_data_type = sub_data_type
            column.nested_columns = None
            return
        elif sub_data_type != column.sub_data_type:
            # Consistent casting can't be done for the column
            column.sub_data_type = None
            column.nested_columns = None
            return

        if sub_data_type.startswith("JSON"):
            _upsert_nested_columns(val, column.name)


def _get_data_type(
    val: Any
) -> Union[str, None]:
    """Infers SQL type from a Python value. Returns TYPE[] for lists, JSON for dicts."""
    if isinstance(val, list):
        inner_types = list({type(x) for x in val})
        if len(inner_types) != 1:
            # Inconsistent type in list prevents proper casting
            return None

        if inner_types[0] == str:
            inner_type = "VARCHAR"
        elif inner_types[0] == dict:
            inner_type = "JSON"
        else:
            inner_type = PYTHON_TO_SQL_TYPE_MAP[inner_types[0]]

        return f"{inner_type}[]"

    elif isinstance(val, dict):
        return "JSON"
    else:
        return None
