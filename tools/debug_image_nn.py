from scraper.embeddings import embed_image_url
from scraper.db import get_engine
from sqlalchemy import text
import numpy as np
import json
import ast
import argparse

# Accept image URL from CLI (positional) or --image; fall back to the old demo image for convenience
parser = argparse.ArgumentParser(description='Debug image nearest-neighbors using an image URL')
parser.add_argument('image_url', nargs='?', help='Image URL to query')
parser.add_argument('--image', dest='image_url_opt', help='Image URL to query (alias)')
args = parser.parse_args()

IMG = args.image_url or args.image_url_opt
if not IMG:
    raise SystemExit('Please provide an image URL: python tools/debug_image_nn.py <image_url>')

print('Computing query embedding for:', IMG)
qvec = embed_image_url(IMG)
qvec = np.asarray(qvec, dtype=float)
qnorm = np.linalg.norm(qvec)
if qnorm > 0:
    qvec = qvec / qnorm
print('query vector shape:', qvec.shape)

engine = get_engine()
with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, title, image_urls, image_embedding FROM products WHERE image_embedding IS NOT NULL")).mappings().all()

print('Fetched', len(rows), 'products with image_embedding')
rows_with_sim = []
for r in rows:
    emb = r.get('image_embedding')
    if not emb:
        continue
    # image_embedding may be stored as a Python list, a JSON string, or a string repr
    if isinstance(emb, str):
        try:
            emb_parsed = json.loads(emb)
        except Exception:
            try:
                emb_parsed = ast.literal_eval(emb)
            except Exception:
                # can't parse; skip
                continue
    else:
        emb_parsed = emb
    arr = np.asarray(emb_parsed, dtype=float)
    if arr.size != qvec.size:
        print('SKIP id', r.get('id'), 'dim mismatch', arr.size)
        continue
    norm = np.linalg.norm(arr)
    if norm == 0:
        continue
    arr = arr / norm
    sim = float(np.dot(qvec, arr))
    rows_with_sim.append((sim, r.get('id'), r.get('title'), r.get('image_urls')))

rows_with_sim.sort(reverse=True, key=lambda x: x[0])

print('\nTop 20 nearest by image-embedding (cosine):')
for sim, pid, title, urls in rows_with_sim[:20]:
    print(f'{sim:.4f}\t{pid}\t{title}\t{urls}')

# quick check: compare with text-based nearest neighbors
from scraper.embeddings import query_similar_by_text
print('\nTop 10 by text similarity for "wireless earbuds":')
text_res = query_similar_by_text('wireless earbuds', top_k=10)
for r in text_res:
    # r may be Row or tuple
    try:
        m = r._mapping
        print(m.get('id'), m.get('title'))
    except Exception:
        print(r[0], r[1])
