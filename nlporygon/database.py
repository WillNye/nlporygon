from typing import Optional

from nlporygon import SupportedDbType
from nlporygon.models import Database


class QueryExecutionError(Exception):
    """Raised when a database query fails to execute."""
    pass


class DuckDb(Database):

    @property
    def database_type(self) -> SupportedDbType:
        return SupportedDbType.DUCK_DB

    async def execute(
        self,
        query: str,
        query_params: Optional[dict] = None,
        **kwargs
    ) -> list[dict]:
        """
        Executes a SQL query and returns results as a list of dictionaries.

        Raises:
            QueryExecutionError: If the query fails or returns invalid results.
        """
        try:
            result = self.connection.execute(
                query, query_params if query_params is not None else {}
            )
        except Exception as e:
            raise QueryExecutionError(f"Query execution failed: {e}") from e

        rows = result.fetchall()

        if not result.description:
            return []

        column_names = [desc[0] for desc in result.description]
        return [dict(zip(column_names, row)) for row in rows]
