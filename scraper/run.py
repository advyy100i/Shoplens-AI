import argparse
from .play_scraper import scrape_url
from .db import get_engine, get_session, Product
from .storage import get_or_create_source, save_product
from .storage import save_review
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def source_from_url(url):
    p = urlparse(url)
    return p.netloc


def main(seeds_file, headless=True):
    engine = get_engine()
    session = get_session(engine)
    with open(seeds_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in tqdm(urls, desc='seeds'):
        src_name = source_from_url(url)
        source = get_or_create_source(session, name=src_name, base_url=src_name)
        data = scrape_url(url, headless=headless)
        product_payload = {
            'source_product_id': None,
            'title': data.get('title'),
            'description': data.get('description'),
            'price': data.get('price'),
            'currency': data.get('currency'),
            'image_urls': data.get('image_urls'),
            'page_url': data.get('page_url')
        }
        save_product(session, source, product_payload)
        # save reviews if found
        try:
            product = session.query(Product).filter_by(page_url=data.get('page_url')).first()
            if product and data.get('reviews'):
                for rv in data.get('reviews'):
                    save_review(session, product, rv)
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', dest='seeds', default='seeds.txt', help='file with seed URLs')
    parser.add_argument('--headless', dest='headless', action='store_true')
    args = parser.parse_args()
    main(args.seeds, headless=args.headless)
