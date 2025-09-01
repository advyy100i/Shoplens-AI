# Audio_Summarization — Data Acquisition (Step 1)

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
