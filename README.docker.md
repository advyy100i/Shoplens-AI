Quick Docker dev setup

Prerequisites
- Docker & Docker Compose installed

Start services

1. Copy example env:

   cp .env.example .env

2. Start the stack:

   docker-compose up --build -d

3. Apply DB migrations / create tables:

   # run a Python one-liner to create tables from SQLAlchemy models
   docker-compose exec web python - <<'PY'
from scraper.db import get_engine, Base
eng = get_engine()
Base.metadata.create_all(eng)
print('tables created')
PY

4. Open the web UI at http://localhost:8000

Notes
- The Postgres service starts with pgvector preloaded via shared_preload_libraries.
- The web service mounts the current directory, so code changes are picked up in the running container.
- For embedding workloads you may want to run them outside the container if you have a GPU on the host.
