# ShopLensAI — Local demo (updated)

Quick demo project that scrapes product pages (Amazon / Flipkart / Myntra) and stores products + reviews in PostgreSQL, computes text & image embeddings (pgvector or JSONB fallback), and exposes simple search endpoints + a tiny UI.

This README includes an up-to-date quick guide to seed 100 products for testing.

## Requirements
- Python 3.9+
- PostgreSQL (with `pgvector` recommended). If using Docker Compose this is automated.
- A virtualenv is recommended

## Quickstart (Windows PowerShell)

1. Activate virtualenv and install requirements
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure Postgres is running and `DATABASE_URL` is set in `.env` (example: `postgresql://postgres:postgres@db:5432/shoplensai`). If using Docker (recommended) see Docker steps below.

3. Seed 100 products (two safe options)

- Fast & safe: synthetic 100 Amazon.in-like products (recommended for development)
```powershell
# create 100 synthetic placeholder products
python scripts/seed_products.py --mode synthetic --count 100
```

- Real product pages (scrape 100 real Amazon.in results — slower, may trigger anti-bot protections)
```powershell
# polite scraping, add --delay to avoid triggers (use proxies for large crawls)
python scripts/seed_products.py --mode scrape --count 100 --delay 1.5
```
Note: Scraping at scale may violate site ToS — use responsibly.

4. Compute embeddings for new products (only unembedded rows)
```powershell
# process only missing embeddings, batched
python -m scraper.embeddings_cli --batch-size 32 --limit 100
```

5. Run the demo API + UI
```powershell
uvicorn scraper.api:app --host 127.0.0.1 --port 8000 --reload
# open http://127.0.0.1:8000/
```

## Docker (optional, recommended for reproducible local dev)
1. Copy `.env.example` → `.env` and adjust as needed.
2. Build & start
```powershell
docker-compose up --build -d
```
3. Create DB tables (once)
```powershell
docker-compose exec web python - <<'PY'
from scraper.db import get_engine, Base
eng = get_engine()
Base.metadata.create_all(eng)
print('tables created')
PY
```
4. Seed and run embeddings inside the container or on host as above.

## Endpoints
- POST /search/text  — body: {"query":"...","top_k":5}
- POST /search/image — body: {"image_url":"...","top_k":5}
- UI at `/` for interactive search and thumbnails

## Notes & caveats
- Synthetic seeder is ideal to quickly populate 100 products for UI/embedding tests.
- Scraping Amazon/Flipkart may be blocked or violate ToS — use proxies, rate-limiting, and obey robots.txt for production crawls.
- If you do not have `pgvector` in Postgres, the project falls back to JSONB storage and Python-side nearest-neighbor as a temporary measure.
- Embedding models can be large — prefer a GPU or run smaller batches on CPU.

## Next suggestions
- Run the embeddings CLI after seeding so image searches are accurate.
- Consider Docker + Docker Compose for reproducible demos.
- For production: add a background worker to compute embeddings asynchronously and a vector index (pgvector ivf/HNSW or FAISS) for scale.