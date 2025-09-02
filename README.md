# ShopLensAI — Local demo

Quick demo project that scrapes product pages (Amazon / Flipkart) and stores products + reviews in PostgreSQL, computes text & image embeddings (pgvector), and exposes simple search endpoints + a tiny UI.

This README gives fast local steps and how to seed 1,000 products for testing.

## Requirements
- Python 3.9+
- PostgreSQL with `pgvector` extension (the project expects a running Postgres and DATABASE_URL set in `.env`)
- A virtualenv is recommended

## Quickstart (Windows PowerShell)

1. Activate virtualenv and install requirements

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure Postgres is running and `DATABASE_URL` is set in `.env` (example: `postgresql://postgres:postgres@localhost:5432/shoplensai`). Create DB and run `scraper/init_db.py` if present.

3. Seed 1,000 products (recommended: synthetic — fast and safe)

```powershell
# create 1000 synthetic Amazon.in-like products
python scripts/seed_products.py --mode synthetic --count 1000
```

If you prefer real product pages (slower, may be blocked) you can run scrape mode — be polite and use `--delay`:

```powershell
# scrape product pages from amazon.in search pages (may trigger blocks)
python scripts/seed_products.py --mode scrape --count 200 --delay 1.5
```

4. Compute embeddings for the newly added products

Run the project's embeddings CLI (example):

```powershell
# compute embeddings for products (this script lives at scraper/embeddings_cli.py)
python scraper/embeddings_cli.py --batch-size 32
```

5. Run the demo API + UI

```powershell
uvicorn scraper.api:app --host 127.0.0.1 --port 8000 --reload
# then open http://127.0.0.1:8000/ to use the UI
```

## Endpoints
- POST /search/text  — body: {"query":"...","top_k":5}
- POST /search/image — body: {"image_url":"...","top_k":5}

The UI at `/` provides simple controls for text and image search.

## Notes and caveats
- The synthetic seeder generates placeholder products (titles, descriptions, images) and is ideal for load testing and UI/embedding development.
- Scraping Amazon/Flipkart at scale may violate their Terms of Service and will likely trigger rate-limiting or blocks — use proxies, IP rotation, and obey robots.txt when crawling.
- Embedding models (sentence-transformers / CLIP) can be CPU and memory heavy. If you have a GPU, the embedding speed improves significantly.
- The project uses PostgreSQL + `pgvector` for vector storage and nearest-neighbor queries. Make sure the extension is installed in your DB.

## Next steps
- Add a background worker for non-blocking scrape+embed jobs.
- Add admin UI for reviewing failed embeddings and re-embedding.
- Add vector index tuning (ivfflat/HNSW) when scaling to >100k vectors.

If you'd like, I can add a one-click script that seeds + embeds + launches the app. Tell me which you'd prefer and I'll implement it.# Audio_Summarization — Data Acquisition (Step 1)

This repository contains the Step 1 implementation: a Playwright-based scraper that collects product data (images, titles, descriptions, prices, and reviews) from e-commerce sites and stores structured data into PostgreSQL.

What you get:
- Playwright scraper with user-agent rotation and basic anti-bot handling
- SQLAlchemy models for `products`, `reviews`, and `sources`
- CLI runner to start scraping a list of seed URLs
- `.env.example` for database configuration

Quick local dev guide
1. Install Python 3.9+ and create a venv
2. pip install -r requirements.txt
3. Install Playwright browsers: `python -m playwright install`
4. Create a PostgreSQL database and enable `pgvector` extension
5. Copy `.env.example` to `.env` and set credentials
6. Run the init script: `python -m scraper.init_db`
7. Run scraper: `python -m scraper.run --seeds seeds.txt`

File structure

```
shoplensai/
├─ scraper/
│  ├─ __init__.py
│  ├─ db.py             # SQLAlchemy models + engine
│  ├─ storage.py        # functions to save products and reviews
│  ├─ play_scraper.py   # Playwright-based scraper
│  ├─ utils.py          # helpers: UA rotation, throttling
│  ├─ init_db.py        # create tables
│  └─ run.py            # CLI runner
├─ requirements.txt
├─ README.md
└─ .env.example
```

Notes
- This step focuses on acquiring and storing data. Future steps will add embedding generation and FastAPI.
- Scraping e-commerce sites like Amazon/Flipkart may violate their ToS — use responsibly and for allowed research.
