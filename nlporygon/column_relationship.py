from collections import defaultdict
from textwrap import dedent
from typing import Union

from pydantic import BaseModel, ConfigDict
from rbloom import Bloom

from nlporygon.logger import logger
from nlporygon.models import Database, Table, TableColumn, ColumnRelationship, Config
from nlporygon.utils import is_date_or_dt_column, format_sql_value, get_query, get_table_name


async def set_column_relationship(
    config: Config,
    db: Database,
    tables: list[Table],
):
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


async def bloom_filter_column_relationship(
    config: Config,
    db: Database,
    tables: list[Table],
):
    column_bloom_map = await create_column_bloom_map(
        config,
        db,
        tables,
    )

    tables = [
        t
        for t in tables
        if t.name in column_bloom_map
    ]

    for primary_idx in range(len(tables)):
        primary_table = tables[primary_idx]

        logger.debug(
            "Detecting undefined foreign key relationships",
            table=primary_table.name,
        )

        primary_bloom_map = column_bloom_map[primary_table.name]
        for p_col_bloom in primary_bloom_map.values():
            if p_col_bloom.total_count != p_col_bloom.unique_value_count:
                # There's non-unique values so it's not a primary key
                continue

            p_col = p_col_bloom.column
            p_bloom = p_col_bloom.bloom

            for secondary_idx in range(len(tables)):
                if primary_idx == secondary_idx:
                    continue  # Skip self-references

                secondary_table = tables[secondary_idx]
                secondary_bloom_map = column_bloom_map[secondary_table.name]
                for sec_col_bloom in secondary_bloom_map.values():
                    sec_col = sec_col_bloom.column

                    if p_col_bloom.column.data_type != sec_col_bloom.column.data_type:
                        continue
                    elif any(
                        r.table == secondary_table.name
                        and r.column == sec_col.name
                        for r in p_col.relationships
                    ):
                        continue
                    elif any(
                        r.table == primary_table.name
                        and r.column == p_col.name
                        for r in sec_col.relationships
                    ):
                        continue
                    elif p_col_bloom.unique_value_count < sec_col_bloom.unique_value_count:
                        continue

                    missing_values = []
                    for val in sec_col_bloom.sample_data:
                        if str(val) not in p_bloom:
                            missing_values.append(val)

                    if missing_values:
                        try:
                            if not await db.execute(
                                dedent(f"""
                                SELECT 1
                                FROM {primary_table.name}
                                WHERE {p_col.query_name} = {format_sql_value(missing_values[0])}
                                LIMIT 1
                                """),
                            ):
                                # Quick check before longer check
                                continue

                            db_results = await db.execute(
                                dedent(f"""
                                SELECT DISTINCT {p_col.query_name}
                                FROM {primary_table.name}
                                WHERE {p_col.query_name} IN {format_sql_value(missing_values)}
                                """),
                            )
                            if len(db_results) != len(missing_values):
                                continue
                        except Exception as e:
                            logger.warning(
                                "Unable to perform column compare",
                                primary_table=primary_table.name,
                                primary_column=p_col.name,
                                child_table=secondary_table.name,
                                child_column=sec_col.name,
                                error=e,
                            )
                            continue

                    sec_col = sec_col_bloom.column
                    sec_col.relationships.append(
                        ColumnRelationship(
                            table=primary_table.name,
                            column=p_col.name
                        )
                    )

