"""
Query persistence and semantic matching using local embeddings.

Stores generated SQL queries with their embeddings for future reuse.
Uses sentence-transformers for embedding generation and FAISS for
efficient in-memory similarity search at scale (100k+ queries).
"""
from datetime import datetime
from typing import Optional

import faiss
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql.sqltypes import Boolean

from nlporygon.logger import logger
from nlporygon.models import Database

Base = declarative_base()


class SavedQueryModel(Base):
    """SQLAlchemy model for persisted queries."""
    __tablename__ = "saved_queries"

    id = Column(Integer, primary_key=True)
    prompt_version = Column(String(255), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    query = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    use_count = Column(Integer, default=0)
    internal = Column(Boolean, nullable=False, default=False)


class SavedQuery(BaseModel):
    """Pydantic model for API responses."""
    id: int
    prompt_version: str
    user_message: str
    query: str
    embedding: list[float]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    internal: bool = False


class QueryStore:
    """
    Stores and retrieves queries using semantic similarity matching.

    Uses sentence-transformers for embedding generation and FAISS for
    fast in-memory similarity search. Embeddings are persisted in the
    database (JSON column) and loaded into a FAISS index on first lookup.

    Performance characteristics:
    - First lookup: O(n) to load embeddings and build FAISS index
    - Subsequent lookups: O(1) via FAISS
    - Memory: ~1.5KB per query (384 floats × 4 bytes)
    """

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings

    def __init__(
        self,
        connection: Database,
        prompt_version: str,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the query store.

        Args:
            connection: SQLAlchemy engine (sync or async) for query storage.
            similarity_threshold: Minimum cosine similarity to consider a match (0-1).
            model_name: Sentence transformer model for embeddings.
        """
        self.db = connection
        self.similarity_threshold = similarity_threshold
        self._model: Optional[SentenceTransformer] = None
        self._model_name = model_name
        self.prompt_version = prompt_version

        # FAISS index state
        self._index: Optional[faiss.IndexFlatIP] = None
        self._index_id_map: dict[int, int] = {}  # faiss_idx -> db_id

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load embedding model on first use."""
        if self._model is None:
            logger.info("Loading embedding model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return self.model.encode(text).tolist()

    async def _build_index(self) -> None:
        """
        Build FAISS index from saved queries for a given prompt version.

        Loads all embeddings from the database and creates an in-memory
        FAISS index for fast similarity search. Uses inner product on
        L2-normalized vectors, which is equivalent to cosine similarity.
        """
        saved = await self._fetch_queries_for_version()

        if not saved:
            self._index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self._index_id_map = {}
            logger.info("Built empty FAISS index", prompt_version=self.prompt_version)
            return

        # Build embedding matrix
        embeddings = np.array([sq.embedding for sq in saved], dtype=np.float32)

        # Normalize for cosine similarity (inner product on unit vectors = cosine)
        faiss.normalize_L2(embeddings)

        # Create index and add embeddings
        self._index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self._index.add(embeddings)

        # Build ID map: faiss index position -> database ID
        self._index_id_map = {i: sq.id for i, sq in enumerate(saved)}

        logger.info("Built FAISS index", num_vectors=len(saved), prompt_version=self.prompt_version)

    async def _fetch_by_id(self, query_id: int) -> Optional[SavedQuery]:
        """Fetch a single saved query by ID."""
        sql = text("""
            SELECT id, prompt_version, user_message, query, embedding,
                   created_at, last_used_at, use_count
            FROM saved_queries
            WHERE id = :id
        """)

        results = await self.db.execute(sql, query_params={"id": query_id})

        if results:
            return SavedQuery(**results[0])
        return None

    async def create_tables(self) -> None:
        """Create the saved_queries table if it doesn't exist."""
        if self.db.is_async:
            async with self.db.connection.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            Base.metadata.create_all(self.db.connection)
        logger.info("Query store tables created")

    async def _fetch_queries_for_version(self) -> list[SavedQuery]:
        """Fetch all saved queries for a given prompt version."""
        query = text("""
            SELECT id, prompt_version, user_message, query, embedding,
                   created_at, last_used_at, use_count
            FROM saved_queries
            WHERE prompt_version = :version
        """)

        result = await self.db.execute(query, {"version": self.prompt_version})

        return [
            SavedQuery(**r)
            for r in result
        ]

    async def _update_usage(self, query_id: int) -> None:
        """Update last_used_at and increment use_count for a query."""
        query = text("""
            UPDATE saved_queries
            SET last_used_at = :now, use_count = use_count + 1
            WHERE id = :id
        """)

        if self.db.is_async:
            async with self.db.connection.begin() as conn:
                await conn.execute(query, {"now": datetime.utcnow(), "id": query_id})
        else:
            with self.db.connection.begin() as conn:
                conn.execute(query, {"now": datetime.utcnow(), "id": query_id})

    async def prune_table(self):
        # Remove queries not being used
        ...

    async def lookup(
        self,
        message: str,
    ) -> Optional[SavedQuery]:
        """
        Find a saved query matching the message above the similarity threshold.

        Uses FAISS for efficient similarity search. On first call for a given
        prompt_version, loads all embeddings and builds the index. Subsequent
        lookups use the in-memory index for O(1) search.

        Args:
            message: The user's natural language query.

        Returns:
            The best matching SavedQuery if similarity >= threshold, else None.
        """
        # Rebuild index if prompt version changed or not initialized
        if self._index is None:
            await self._build_index(self.prompt_version)

        if self._index.ntotal == 0:
            return None

        # Get query embedding and normalize for cosine similarity
        query_embedding = np.array([self.get_embedding(message)], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search FAISS index (k=1 for best match)
        similarities, indices = self._index.search(query_embedding, k=1)
        similarity = float(similarities[0][0])
        faiss_idx = int(indices[0][0])

        if similarity < self.similarity_threshold:
            return None

        # Fetch full record from DB
        db_id = self._index_id_map[faiss_idx]
        saved_query = await self._fetch_by_id(db_id)

        if saved_query:
            logger.info(
                "Found matching saved query",
                similarity=round(similarity, 3),
                saved_message=saved_query.user_message[:50]
            )
            await self._update_usage(db_id)

        return saved_query

    async def save(
        self,
        user_message: str,
        query: str
    ) -> SavedQuery:
        """
        Save a query with its embedding for future reuse.

        Also updates the in-memory FAISS index if it's initialized for
        this prompt version, avoiding the need to rebuild the full index.

        Args:
            user_message: The original natural language query.
            query: The generated SQL query.

        Returns:
            The saved query object.
        """
        embedding = self.get_embedding(user_message)
        now = datetime.now()

        insert_query = text("""
            INSERT INTO saved_queries
                (prompt_version, user_message, query, embedding, created_at, use_count)
            VALUES
                (:prompt_version, :user_message, :query, :embedding, :created_at, 0)
            RETURNING id
        """)

        params = {
            "prompt_version": self.prompt_version,
            "user_message": user_message,
            "query": query,
            "embedding": embedding,
            "created_at": now,
        }

        response = await self.db.execute(insert_query, query_params=params)
        query_id = response[0]["id"]

        logger.info("Saved query to store", user_message=user_message[:50])

        saved_query = SavedQuery(
            id=query_id,
            prompt_version=self.prompt_version,
            user_message=user_message,
            query=query,
            embedding=embedding,
            created_at=now,
            use_count=0
        )

        # Update in-memory FAISS index if initialized for this version
        if self._index is not None:
            embedding_arr = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_arr)
            self._index.add(embedding_arr)

            faiss_idx = self._index.ntotal - 1
            self._index_id_map[faiss_idx] = query_id

        return saved_query
