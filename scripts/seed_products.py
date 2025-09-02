#!/usr/bin/env python3
"""Seed the products table with many entries.

Usage:
  python scripts/seed_products.py --mode synthetic --count 1000
  python scripts/seed_products.py --mode scrape --count 1000

Modes:
  synthetic - create varied placeholder products (fast, safe)
  scrape    - try to crawl Amazon.in search pages and scrape product pages (slow, may trigger blocks)

This script uses the project's DB layer and `storage.save_product` to persist rows.
"""
import argparse
import random
import time
from urllib.parse import urljoin

from scraper.db import get_engine, get_session
from scraper.storage import get_or_create_source, save_product

LOREM = (
    "Lightweight, durable, and designed for everyday use. Perfect for commuters and students. "
    "Water-resistant exterior with padded laptop compartment and multiple pockets."
)

AMAZON_SEARCH_PAGES = [
    # some category/search pages on amazon.in to seed scraping mode
    "https://www.amazon.in/s?k=laptop",
    "https://www.amazon.in/s?k=mobile+phone",
    "https://www.amazon.in/s?k=backpack",
    "https://www.amazon.in/s?k=shoes",
    "https://www.amazon.in/s?k=headphones",
    "https://www.amazon.in/s?k=watch",
]


def seed_synthetic(count: int = 1000):
    engine = get_engine()
    session = get_session(engine)
    src = get_or_create_source(session, name='Amazon India (synthetic)', base_url='https://www.amazon.in')
    created = 0
    for i in range(1, count + 1):
        title = random.choice(['Ultra', 'Pro', 'Max', 'Lite', 'Classic']) + f' Product {i}'
        cat = random.choice(['Laptop', 'Backpack', 'Shoes', 'Headphones', 'Watch', 'Phone', 'Bag'])
        title = f'{title} - {cat}'
        desc = LOREM + f" Specs: {random.randint(1,16)}GB RAM, {random.randint(128,2048)}GB storage."
        price = round(random.uniform(499, 129999), 2)
        img = f'https://picsum.photos/seed/amazon_synth_{i}/640/480'
        product_data = {
            'source_product_id': f'SYNTH-{i}',
            'title': title,
            'description': desc,
            'price': price,
            'currency': 'INR',
            'image_urls': [img],
            'page_url': f'https://www.amazon.in/dp/SYNTH{i}'
        }
        save_product(session, src, product_data)
        created += 1
        if created % 100 == 0:
            print(f'Created {created} synthetic products')

    print(f'Done: created {created} synthetic Amazon.in products')


def collect_links_from_search(html):
    # lightweight extraction of product links from Amazon search page HTML
    # This is best-effort and uses simple heuristics. Returns absolute URLs.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    # Amazon places results under div.s-main-slot; links often have class 'a-link-normal s-no-outline'
    for a in soup.select('a.a-link-normal.s-no-outline'):
        href = a.get('href')
        if href and '/dp/' in href:
            links.append(urljoin('https://www.amazon.in', href.split('?')[0]))
    # fallback: any /dp/ links
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and '/dp/' in href:
            full = urljoin('https://www.amazon.in', href.split('?')[0])
            if full not in links:
                links.append(full)
    return links


def seed_by_scrape(count: int = 1000, delay: float = 1.0):
    import requests
    from scraper.play_scraper import scrape_url

    engine = get_engine()
    session = get_session(engine)
    src = get_or_create_source(session, name='Amazon India (scraped)', base_url='https://www.amazon.in')

    seen = set()
    to_scrape = []

    # Prefer using Playwright-rendered pages to collect links (more reliable on Amazon)
    try:
        from playwright.sync_api import sync_playwright
        print('Using Playwright to collect search result links')
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent='Mozilla/5.0')
            page = context.new_page()
            for s in AMAZON_SEARCH_PAGES:
                try:
                    page.goto(s, timeout=30000)
                    page.wait_for_timeout(1500)
                    html = page.content()
                    links = collect_links_from_search(html)
                    for l in links:
                        if l not in seen:
                            to_scrape.append(l)
                            seen.add(l)
                        if len(to_scrape) >= count:
                            break
                except Exception as e:
                    print('playwright search fetch error', e)
                if len(to_scrape) >= count:
                    break
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
    except Exception:
        print('Playwright not available or failed, falling back to requests for search pages')
        for page in AMAZON_SEARCH_PAGES:
            try:
                r = requests.get(page, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                links = collect_links_from_search(r.text)
                for l in links:
                    if l not in seen:
                        to_scrape.append(l)
                        seen.add(l)
                    if len(to_scrape) >= count:
                        break
            except Exception as e:
                print('search fetch error', e)
            if len(to_scrape) >= count:
                break

    print(f'Collected {len(to_scrape)} product links to scrape (will scrape up to {count})')

    scraped = 0
    for url in to_scrape[:count]:
        try:
            data = scrape_url(url)
            product_data = {
                'source_product_id': (url.split('/dp/')[-1] or url).split('/')[0],
                'title': data.get('title'),
                'description': data.get('description'),
                'price': data.get('price'),
                'currency': data.get('currency'),
                'image_urls': data.get('image_urls') or [],
                'page_url': data.get('page_url') or url,
            }
            save_product(session, src, product_data)
            scraped += 1
            if scraped % 20 == 0:
                print(f'Scraped and saved {scraped} products')
        except Exception as e:
            print('scrape error for', url, e)
        time.sleep(delay)

    print(f'Done: scraped and saved {scraped} products')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['synthetic', 'scrape'], default='synthetic')
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--delay', type=float, default=1.0, help='delay between requests in scrape mode')
    args = p.parse_args()
    if args.mode == 'synthetic':
        seed_synthetic(args.count)
    else:
        seed_by_scrape(args.count, delay=args.delay)


if __name__ == '__main__':
    main()
