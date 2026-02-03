from typing import Optional

from nlporygon import SupportedDbType
from nlporygon.models import Database


class DuckDb(Database):

    @property
    def database_type(self) -> SupportedDbType:
        return SupportedDbType.DUCK_DB

    async def execute(
        self,
        query: str,
        query_params: Optional[dict] = None,
        **kwargs
    ):
        q = self.connection.execute(
            query, query_params if query_params is not None else {}
        )
        rows = q.fetchall()
        assert q.description
        column_names = [desc[0] for desc in q.description]
        return [dict(zip(column_names, row)) for row in rows]
