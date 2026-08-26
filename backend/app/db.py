from typing import Optional

import asyncpg

from .config import DATABASE_URL

_pool: Optional[asyncpg.Pool] = None


def _needs_ssl(dsn: str) -> bool:
    # Neon (and most managed Postgres) require TLS; local dev usually doesn't.
    return "localhost" not in dsn and "127.0.0.1" not in dsn


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                dsn=DATABASE_URL,
                ssl="require" if _needs_ssl(DATABASE_URL) else None,
                min_size=1,
                max_size=5,
            )
            async with _pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        gender TEXT,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
        except Exception as e:
            print(f"\n❌ [DATABASE CONNECTION ERROR] Failed to connect to PostgreSQL.")
            print(f"Current DATABASE_URL: {DATABASE_URL}")
            print("Please create/update 'backend/.env' with your PostgreSQL connection string.")
            print("Example: DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres\n")
            raise RuntimeError(f"Could not connect to database: {e}") from e
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized — call init_pool() on startup.")
    return _pool
