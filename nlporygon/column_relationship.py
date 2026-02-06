"""
Discovers relationships between database columns using two approaches:

1. Explicit foreign keys: Queries database metadata for declared FK constraints.
2. Bloom filter heuristics: Detects implicit relationships where FKs aren't defined
   by using bloom filters to efficiently test if one column's values are a subset
   of another's (indicating a potential FK→PK relationship).
"""
from collections import defaultdict
from textwrap import dedent
from typing import Union

from pydantic import BaseModel, ConfigDict
from rbloom import Bloom
from sqlalchemy.exc import SQLAlchemyError

from nlporygon.logger import logger
from nlporygon.schema import Table, TableColumn, ColumnRelationship
from nlporygon.database import Database, get_query, is_date_or_dt_column, format_sql_value, get_table_name
from nlporygon.config import Config


async def set_column_relationship(
    config: Config,
    db: Database,
    tables: list[Table],
):
    """
    Populates column.relationships for all tables by querying explicit FK constraints
    from the database, then runs bloom filter heuristics to detect implicit relationships.
    """
    logger.info("Adding table definition foreign key relationships")
    results = await db.execute(
        get_query("get_fks.sql", db.database_type)
    )
    relationship_map = defaultdict(lambda: defaultdict(list))
    for row in results:
        if not row["referenced_table"] or not row["referenced_column"]:
            continue

        relationship_map[get_table_name(row)][row["column_name"]].append(
            ColumnRelationship(
                table=get_table_name(row, "referenced_"),
                column=row["referenced_column"],
            )
        )

    for table in tables:
        for column in table.columns:
            relationships = relationship_map[table.name][column.name]
            for relationship in relationships:
                if not any(
                    relationship.table == r.table
                    and relationship.column == r.column
                    for r in column.relationships
                ):
                    column.relationships.append(relationship)

    await bloom_filter_column_relationship(config, db, tables)


class ColumnBloom(BaseModel):
    """
    Wraps a column with its bloom filter and cardinality stats.

    The total_count vs unique_value_count comparison identifies potential primary keys:
    columns where every value is unique can be the "one" side of a one-to-many relationship.
    """
    column: TableColumn
    bloom: Bloom
    sample_data: list
    total_count: int
    unique_value_count: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def name(self) -> str:
        return self.column.name


async def create_bloom_filter(
    config: Config,
    db: Database,
    table: Table,
    column: TableColumn,
) -> Union[ColumnBloom, None]:
    """
    Builds a ColumnBloom from distinct values in a column.
    Returns None if the column is empty or contains unhashable types.
    """
    limit = config.column_relationships.bloom_size
    bloom = Bloom(limit, config.column_relationships.error_rate)
    sample_data = []
    db_data = await db.execute(dedent(f"""
    SELECT DISTINCT {column.query_name}
    FROM {table.name}
    WHERE {column.query_name} IS NOT NULL
    ORDER BY {column.query_name}
    LIMIT {limit}
    """))

    for row in db_data[:config.column_relationships.sample_size]:
        sample_data.append(row[column.name])

    for row in db_data:
        try:
            bloom.add(str(row[column.name]))
        except TypeError:
            return None

    unique_value_count = await db.execute(dedent(f"""
    SELECT COUNT(DISTINCT {column.query_name}) AS col_count
    FROM {table.name}
    WHERE {column.query_name} IS NOT NULL
    """))

    if not unique_value_count[0]["col_count"]:
        logger.info(
            "No data found for column",
            table=table.name,
            column=column.name,
        )
        return None

    total_count = await db.execute(dedent(f"""
    SELECT COUNT({column.query_name}) AS col_count
    FROM {table.name}
    WHERE {column.query_name} IS NOT NULL
    """))

    return ColumnBloom(
        column=column,
        bloom=bloom,
        sample_data=sample_data,
        total_count=total_count[0]["col_count"],
        unique_value_count=unique_value_count[0]["col_count"]
    )


async def create_column_bloom_map(
    config: Config,
    db: Database,
    tables: list[Table],
) -> dict[str, dict[str, ColumnBloom]]:
    """
    Creates a ColumnBloom for every eligible column across all tables.
    Skips types unsuitable for join relationships (dates, JSON, arrays, booleans, enums).
    """
    response: dict[str, dict[str, ColumnBloom]] = {}
    logger.info("Creating bloom map for all columns across all tables")

    for table in tables:
        table_check = await db.execute(f"SELECT * FROM {table.name} LIMIT 1")
        if not table_check:
            logger.info(
                "No data found for table",
                table=table.name,
            )
            continue

        logger.debug(
            f"Creating bloom filter and retrieving sample data",
            table=table.name
        )
        response[table.name] = {}
        for column in table.columns:
            if column.name in config.column_relationships.global_ignore_column_rules:
                continue
            elif is_date_or_dt_column(column.data_type):
                continue
            elif "JSON" in column.data_type:
                continue
            elif "[]" in column.data_type:
                continue
            elif column.data_type == "BOOLEAN":
                continue
            elif column.data_type.startswith("ENUM"):
                continue

            bloom_filter = await create_bloom_filter(
                config, db, table, column,
            )
            if bloom_filter:
                response[table.name][column.name] = bloom_filter

    logger.info("Bloom map creation complete")
    return response


def _relationship_exists(
    p_col: TableColumn,
    sec_col: TableColumn,
    primary_table: str,
    secondary_table: str
) -> bool:
    """Returns True if relationship already exists in either direction."""
    forward = any(
        r.table == secondary_table and r.column == sec_col.name
        for r in p_col.relationships
    )
    reverse = any(
        r.table == primary_table and r.column == p_col.name
        for r in sec_col.relationships
    )
    return forward or reverse


def _is_valid_relationship_candidate(
    p_col_bloom: ColumnBloom,
    sec_col_bloom: ColumnBloom,
    primary_table: str,
    secondary_table: str
) -> bool:
    """Returns True if columns could have a FK→PK relationship."""
    if p_col_bloom.column.data_type != sec_col_bloom.column.data_type:
        return False

    if _relationship_exists(
        p_col_bloom.column, sec_col_bloom.column,
        primary_table, secondary_table
    ):
        return False

    if p_col_bloom.unique_value_count < sec_col_bloom.unique_value_count:
        return False

    return True


async def _verify_missing_values(
    db: Database,
    primary_table: str,
    p_col: TableColumn,
    missing_values: list
) -> bool:
    """
    Verifies that values missing from bloom filter actually exist in primary table.
    Returns True if all missing values are found (bloom false positives).
    """
    if not missing_values:
        return True

    try:
        # Quick single-value check first
        exists = await db.execute(dedent(f"""
            SELECT 1
            FROM {primary_table}
            WHERE {p_col.query_name} = {format_sql_value(missing_values[0])}
            LIMIT 1
        """))
        if not exists:
            return False

        # Full check for all missing values
        db_results = await db.execute(dedent(f"""
            SELECT DISTINCT {p_col.query_name}
            FROM {primary_table}
            WHERE {p_col.query_name} IN {format_sql_value(missing_values)}
        """))
        return len(db_results) == len(missing_values)

    except SQLAlchemyError as e:
        logger.warning(
            "Unable to verify missing values",
            table=primary_table,
            column=p_col.name,
            error=str(e),
        )
        return False


async def _find_relationships_for_column(
    db: Database,
    tables: list[Table],
    column_bloom_map: dict[str, dict[str, ColumnBloom]],
    primary_table: Table,
    p_col_bloom: ColumnBloom,
):
    """Finds all FK relationships pointing to the given primary column."""
    p_col = p_col_bloom.column
    p_bloom = p_col_bloom.bloom

    for secondary_table in tables:
        if secondary_table.name == primary_table.name:
            continue

        secondary_bloom_map = column_bloom_map[secondary_table.name]
        for sec_col_bloom in secondary_bloom_map.values():
            if not _is_valid_relationship_candidate(
                p_col_bloom, sec_col_bloom,
                primary_table.name, secondary_table.name
            ):
                continue

            missing_values = [
                val for val in sec_col_bloom.sample_data
                if str(val) not in p_bloom
            ]
            if not await _verify_missing_values(db, primary_table.name, p_col, missing_values):
                continue

            # Add relationship: FK column → PK column
            sec_col_bloom.column.relationships.append(
                ColumnRelationship(
                    table=primary_table.name,
                    column=p_col.name
                )
            )


async def bloom_filter_column_relationship(
    config: Config,
    db: Database,
    tables: list[Table],
):
    """
    Detects implicit FK relationships using bloom filters.

    For each column with 100% unique values (potential PK), checks if other columns'
    values are subsets of it. Uses bloom filters for fast "definitely not present"
    checks, then verifies bloom "maybe present" results with actual DB queries.
    Relationships are added from the secondary (FK) column pointing to the primary (PK).
    """
    column_bloom_map = await create_column_bloom_map(config, db, tables)
    tables = [t for t in tables if t.name in column_bloom_map]

    for primary_table in tables:
        logger.debug("Detecting undefined FK relationships", table=primary_table.name)
        primary_bloom_map = column_bloom_map[primary_table.name]

        for p_col_bloom in primary_bloom_map.values():
            if not p_col_bloom.total_count == p_col_bloom.unique_value_count:
                continue

            await _find_relationships_for_column(
                db, tables, column_bloom_map,
                primary_table, p_col_bloom
            )

