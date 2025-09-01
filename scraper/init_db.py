from .db import get_engine, Base
from sqlalchemy import text


def init_db(create_pgvector: bool = False):
    engine = get_engine(echo=True)
    Base.metadata.create_all(engine)
    if create_pgvector:
        # Attempt to enable pgvector extension (requires privileges)
        with engine.connect() as conn:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            conn.commit()


if __name__ == '__main__':
    # default: do not attempt to create extension
    init_db(create_pgvector=False)
