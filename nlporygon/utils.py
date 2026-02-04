import functools
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

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
