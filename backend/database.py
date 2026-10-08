import os
from uuid import uuid4
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    migrate_chat_schema(engine)


def migrate_chat_schema(bind) -> None:
    """Add status to existing installations without rewriting message history."""
    changes = [('chat_messages', 'status', "VARCHAR(20) NOT NULL DEFAULT 'complete'"),
               ('chat_turns', 'parent_request_id', 'VARCHAR(36)'),
               ('chat_sessions', 'instance_id', "VARCHAR(36) NOT NULL DEFAULT ''")]
    for table, name, definition in changes:
        inspector = inspect(bind)
        if inspector.has_table(table) and name not in {column['name'] for column in inspector.get_columns(table)}:
            with bind.begin() as connection:
                connection.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {definition}'))
    if inspect(bind).has_table('chat_sessions'):
        with bind.begin() as connection:
            rows = connection.execute(text("SELECT id FROM chat_sessions WHERE instance_id = ''")).all()
            for row in rows:
                connection.execute(text('UPDATE chat_sessions SET instance_id = :nonce WHERE id = :id'),
                                   {'nonce': str(uuid4()), 'id': row.id})
