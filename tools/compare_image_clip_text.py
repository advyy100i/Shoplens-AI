import os
import json
import ast
import numpy as np
from scraper.embeddings import embed_image_url, embed_text_with_clip
from scraper.db import get_engine
from sqlalchemy import text

import argparse

DEFAULT_TOP_K = 20

parser = argparse.ArgumentParser(description='Compare image NN vs CLIP-text NN for a query image')
parser.add_argument('--image-url', '-i', required=True, help='Query image URL')
parser.add_argument('--top-k', '-k', type=int, default=DEFAULT_TOP_K, help='How many top neighbors to show')
args = parser.parse_args()

IMG = args.image_url
TOP_K = args.top_k

print('Computing query image embedding for:', IMG)
qvec = embed_image_url(IMG)
qvec = np.asarray(qvec, dtype=float)
qnorm = np.linalg.norm(qvec)
if qnorm > 0:
    qvec = qvec / qnorm
print('query vector dim:', qvec.shape)

engine = get_engine()
with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, title, image_urls, image_embedding FROM products WHERE image_embedding IS NOT NULL"))
    rows = rows.mappings().all()

print('Loaded', len(rows), 'products with image_embedding')

# parse embeddings and prepare lists
ids = []
titles = []
img_embs = []
img_urls = []
for r in rows:
    emb = r.get('image_embedding')
    if not emb:
        continue
    if isinstance(emb, str):
        try:
            emb_parsed = json.loads(emb)
        except Exception:
            try:
                emb_parsed = ast.literal_eval(emb)
            except Exception:
                continue
    else:
        emb_parsed = emb
    arr = np.asarray(emb_parsed, dtype=float)
    if arr.size != qvec.size:
        continue
    norm = np.linalg.norm(arr)
    if norm == 0:
        continue
    arr = arr / norm
    ids.append(r.get('id'))
    titles.append(r.get('title') or '')
    img_embs.append(arr)
    img_urls.append(r.get('image_urls'))

img_embs = np.stack(img_embs)
print('Prepared', img_embs.shape[0], 'normalized image embeddings')

# Compute CLIP text embeddings for the titles (in batches)
print('Computing CLIP text embeddings for titles...')
clip_texts = embed_text_with_clip(titles, batch_size=64)
print('CLIP text embeddings shape:', clip_texts.shape)

# compute cosine similarities
sim_image = img_embs.dot(qvec)
sim_clip_text = clip_texts.dot(qvec)

# get top-K indices
top_img_idx = np.argsort(-sim_image)[:TOP_K]
top_clip_idx = np.argsort(-sim_clip_text)[:TOP_K]

print('\nTop image-embedding nearest products:')
for rank, idx in enumerate(top_img_idx, start=1):
    print(f'{rank:2d}. id={ids[idx]} sim={sim_image[idx]:.4f} title={titles[idx]} url={img_urls[idx]}')

print('\nTop CLIP-text-embedding nearest products:')
for rank, idx in enumerate(top_clip_idx, start=1):
    sim_clip_val = sim_clip_text[idx]
    print(f'{rank:2d}. id={ids[idx]} sim={sim_clip_val:.4f} title={titles[idx]} url={img_urls[idx]}')

# compute overlap
img_set = set(ids[i] for i in top_img_idx)
clip_set = set(ids[i] for i in top_clip_idx)
common = img_set & clip_set
print(f"\nTop-{TOP_K} overlap: {len(common)} items ({len(common)/TOP_K*100:.1f}%)")
print('Common ids:', sorted(common))
