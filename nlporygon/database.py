import functools
from pathlib import Path
from typing import Union, Optional, Any

from jinja2 import Template
from pydantic import BaseModel
from sqlalchemy import Engine, text
from sqlalchemy.ext.asyncio import AsyncEngine


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

    @property
    def is_async(self) -> bool:
        return isinstance(self.connection, AsyncEngine)

    async def execute(
        self,
        query: str,
        query_params: Optional[dict] = None,
        commit: bool = False,
        **kwargs
    ) -> list[dict]:
        """
        Executes a SQL query and returns results as a list of dictionaries.

        Args:
            query: SQL query string. Use :param_name for parameter binding.
            query_params: Optional dict of parameter values.
            commit: Call commit after executing the query.

        Returns:
            List of dicts, one per row, with column names as keys.
        """
        if self.is_async:
            async with self.connection.connect() as conn:
                result = await conn.execute(
                    text(query),
                    query_params or {}
                )
                rows = result.fetchall()
                if commit:
                    await conn.commit()
        else:
            with self.connection.connect() as conn:
                result = conn.execute(
                    text(query),
                    query_params or {}
                )
                rows = result.fetchall()
                if commit:
                    conn.commit()

        if not result.keys():
            return []

        column_names = list(result.keys())
        return [dict(zip(column_names, row)) for row in rows]


QUERY_DIR = Path(__file__).parent / "queries"


@functools.cache
def _get_query(
    query_name: str,
    db_type: str
) -> str:
    if not query_name.endswith('.sql'):
        query_name += '.sql'
    if QUERY_DIR.joinpath(db_type, query_name).exists():
        return QUERY_DIR.joinpath(db_type, query_name).read_text()
    else:
        return QUERY_DIR.joinpath(query_name).read_text()


def get_query(
    query_name: str,
    db_type: str
) -> str:
    return _get_query(query_name, db_type)


def get_query_template(
    query_name: str,
    db_type: str
):
    return Template(
        _get_query(query_name, db_type)
    )


def is_date_or_dt_column(data_type: str) -> bool:
    data_type = data_type.upper()
    return any(t in data_type for t in [
            "DATE",
            "TIME",
            "INTERVAL",
       ]
    )


def format_sql_value(val):
    """Format value for SQL.
    It's not pretty, but it maximizes compatability
    """
    if val is None:
        return 'NULL'
    elif isinstance(val, list) or isinstance(val, set):
        val = ','.join(format_sql_value(v) for v in val)
        return f"({val})"
    elif isinstance(val, (int, float)):
        return str(val)
    else:
        val = str(val)
        return f"'{val.replace("'", "''")}'"


def get_table_name(
    row_dict: dict[str, Any],
    column_prefix: Optional[str] = None
) -> str:
    column_prefix = column_prefix or ''
    table_catalog = f'"{row_dict[f"{column_prefix}table_catalog"]}"'
    table_schema = f'"{row_dict[f"{column_prefix}table_schema"]}"'
    table_name = f'"{row_dict[f"{column_prefix}table_name"]}"'
    return f"{table_catalog}.{table_schema}.{table_name}"
