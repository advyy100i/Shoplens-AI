import argparse
from .embeddings import ensure_vector_columns, embed_texts, embed_image_url, store_product_text_embedding, store_product_image_embedding
from .db import get_engine, get_session, Product


def run_generate(all_products=False, text_only=False, image_only=False, limit=100):
    ensure_vector_columns()
    session = get_session()
    qs = session.query(Product)
    if not all_products:
        qs = qs.limit(limit)
    for p in qs.all():
        if not image_only and (p.title or p.description):
            txt = (p.title or '') + '\n' + (p.description or '')
            v = embed_texts([txt])[0]
            store_product_text_embedding(p.id, v)
        if not text_only and p.image_urls:
            # take first image
            try:
                v = embed_image_url(p.image_urls[0])
                store_product_image_embedding(p.id, v)
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--image-only', action='store_true')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    run_generate(all_products=args.all, text_only=args.text_only, image_only=args.image_only, limit=args.limit)
