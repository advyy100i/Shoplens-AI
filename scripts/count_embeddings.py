from scraper.db import get_engine
from sqlalchemy import text

def main():
    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text('SELECT COUNT(*) FROM products')).scalar()
        text_cnt = conn.execute(text('SELECT COUNT(*) FROM products WHERE product_embedding IS NOT NULL')).scalar()
        img_cnt = conn.execute(text('SELECT COUNT(*) FROM products WHERE image_embedding IS NOT NULL')).scalar()
    print(f"total_products={total}, text_embeddings={text_cnt}, image_embeddings={img_cnt}")

if __name__ == '__main__':
    main()
