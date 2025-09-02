"""Run the ShopLensAI demo API.

Usage:
    python run_api.py
"""
import os

if __name__ == '__main__':
    # run uvicorn programmatically to keep things simple
    import uvicorn
    uvicorn.run('scraper.api:app', host='127.0.0.1', port=8000, log_level='info')
