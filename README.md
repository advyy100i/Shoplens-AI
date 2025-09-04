
# ShopLensAI

**ShopLensAI** is an intelligent product search and recommendation engine.
It scrapes product data from major e-commerce sites (Amazon, Flipkart, Myntra), stores products and reviews in PostgreSQL, generates embeddings for both text and images, and provides semantic search through a lightweight API and UI.

---

## Features

* **Data ingestion**

  * Scrape product pages (titles, descriptions, prices, reviews, images).
  * Synthetic data generator for quick local development.

* **Storage**

  * PostgreSQL as the primary database.
  * Native vector search with `pgvector`.
  * Fallback to JSONB + Python nearest-neighbor if `pgvector` is unavailable.

* **Embeddings**

  * Text and image embeddings for semantic search.
  * Batched processing to handle large datasets efficiently.

* **Search API**

  * Text-based semantic search.
  * Image similarity search.
  * REST endpoints for integration with other systems.

* **Web UI**

  * Minimal interface for interactive product search with thumbnails.

---

## Requirements

* Python **3.9+**
* PostgreSQL (**pgvector** extension recommended)
* Virtual environment (`venv`) suggested
* Optional: **Docker + Docker Compose** for reproducible development

---

## Installation (Windows PowerShell)

1. **Set up environment**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Configure database**
   Ensure PostgreSQL is running and `DATABASE_URL` is set in `.env`

   ```
   postgresql://postgres:postgres@db:5432/shoplensai
   ```

3. **Create database tables**

```powershell
python - <<'PY'
from scraper.db import get_engine, Base
eng = get_engine()
Base.metadata.create_all(eng)
print("tables created")
PY
```

---

## Data Seeding

Two supported modes:

* **Synthetic dataset (recommended for development)**

```powershell
python scripts/seed_products.py --mode synthetic --count 100
```

* **Real product scraping (Amazon.in)**
  *(use proxies and delays to avoid blocks; respect ToS)*

```powershell
python scripts/seed_products.py --mode scrape --count 100 --delay 1.5
```

---

## Embedding Generation

Generate embeddings for products that don’t yet have them:

```powershell
python -m scraper.embeddings_cli --batch-size 32 --limit 100
```

---

## Running the Application

Start API + UI locally:

```powershell
uvicorn scraper.api:app --host 127.0.0.1 --port 8000 --reload
```

Access UI at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## Docker Setup (Recommended)

1. Copy `.env.example` → `.env` and configure.
2. Build and start services:

```powershell
docker-compose up --build -d
```

3. Initialize database tables:

```powershell
docker-compose exec web python - <<'PY'
from scraper.db import get_engine, Base
eng = get_engine()
Base.metadata.create_all(eng)
print("tables created")
PY
```

---

## API Endpoints

* **POST** `/search/text`
  Body: `{"query": "laptop bag", "top_k": 5}`

* **POST** `/search/image`
  Body: `{"image_url": "https://...jpg", "top_k": 5}`

* **GET** `/`
  Interactive UI for search and browsing results.

---

## Notes

* Use **synthetic data** for rapid development and testing.
* Real scraping may trigger anti-bot measures; use responsibly with rate limits and proxies.
* Embedding models can be resource-intensive. Prefer GPUs or small batches for CPU execution.
---
<img width="1299" height="732" alt="Screenshot 2025-09-02 033648" src="https://github.com/user-attachments/assets/99812360-ec37-4f40-88f9-437582510865" />

