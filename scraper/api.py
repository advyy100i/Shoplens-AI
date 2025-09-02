from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from .embeddings import query_similar_by_text, query_similar_by_image_url, embed_image_url, embed_image_bytes
from .embeddings import embed_text_with_clip, ensure_clip_text_column
from .db import get_engine
import numpy as np
import json, ast

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi.responses import JSONResponse
from fastapi import Request
import traceback
from fastapi import UploadFile, File
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = FastAPI(title='ShopLensAI Demo API')

# Serve static files under /static to avoid masking API routes. Use absolute path.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')
app.mount('/static', StaticFiles(directory=WEB_DIR), name='web')


@app.get('/', include_in_schema=False)
def root_ui():
    index_path = os.path.join(WEB_DIR, 'index.html')
    return FileResponse(index_path)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    # return a compact error JSON; full traceback is logged to console
    print(tb)
    return JSONResponse(status_code=500, content={'error': str(exc), 'type': type(exc).__name__})


class TextQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5


class ImageQuery(BaseModel):
    image_url: str
    top_k: Optional[int] = 5



@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/search/text')
def search_text(q: TextQuery):
    rows = query_similar_by_text(q.query, top_k=q.top_k)
    # convert SQLAlchemy Row objects to plain dicts
    out = []
    for r in rows:
        # SQLAlchemy Row can be accessed as a mapping via _mapping or as a tuple
        if hasattr(r, '_mapping'):
            m = r._mapping
            id_ = m.get('id')
            title = m.get('title')
            page_url = m.get('page_url')
            image_urls = m.get('image_urls') or []
            dist = float(m.get('dist'))
        else:
            # fallback to positional indexing
            id_ = r[0]
            title = r[1]
            page_url = r[2]
            image_urls = r[3] if len(r) > 3 and r[3] is not None else []
            dist = float(r[-1])
        out.append({'id': id_, 'title': title, 'page_url': page_url, 'image_urls': image_urls, 'dist': dist})
    return {'results': out}


@app.post('/search/image')
def search_image(q: ImageQuery):
    # Try image-based nearest neighbors first
    rows = query_similar_by_image_url(q.image_url, top_k=q.top_k)
    out = []
    if rows and len(rows) > 0:
        for r in rows:
            if hasattr(r, '_mapping'):
                m = r._mapping
                id_ = m.get('id')
                title = m.get('title')
                page_url = m.get('page_url')
                image_urls = m.get('image_urls') or []
                dist = float(m.get('dist'))
            else:
                id_ = r[0]
                title = r[1]
                page_url = r[2]
                image_urls = r[3] if len(r) > 3 and r[3] is not None else []
                dist = float(r[-1])
            out.append({'id': id_, 'title': title, 'page_url': page_url, 'image_urls': image_urls, 'dist': dist})
        return {'results': out}

    # Fallback: if image-NN returned nothing, compute CLIP-text similarity over stored clip_text_embedding
    try:
        ensure_clip_text_column()
        qvec = embed_image_url(q.image_url)
        qvec = np.asarray(qvec, dtype=float)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute("SELECT id, title, page_url, image_urls, clip_text_embedding FROM products WHERE clip_text_embedding IS NOT NULL").mappings().all()

        sims = []
        if rows:
            # Use stored clip_text_embedding values
            for r in rows:
                emb = r.get('clip_text_embedding')
                if not emb:
                    continue
                # parse if string
                if isinstance(emb, str):
                    try:
                        emb_p = json.loads(emb)
                    except Exception:
                        try:
                            emb_p = ast.literal_eval(emb)
                        except Exception:
                            continue
                else:
                    emb_p = emb
                arr = np.asarray(emb_p, dtype=float)
                if arr.size != qvec.size:
                    continue
                norm = np.linalg.norm(arr)
                if norm == 0:
                    continue
                arr = arr / norm
                sim = float(np.dot(qvec, arr))
                sims.append((sim, r))
        else:
            # No precomputed CLIP text embeddings found — compute CLIP text embeddings on-the-fly
            # This is slower but ensures the image search returns useful results even before batch precomputation.
            with engine.connect() as conn:
                all_rows = conn.execute("SELECT id, title, page_url, image_urls FROM products WHERE title IS NOT NULL").mappings().all()

            # Batch titles to avoid OOM for large DBs
            batch_size = 64
            ids = [r.get('id') for r in all_rows]
            titles = [r.get('title') or '' for r in all_rows]
            from .embeddings import embed_text_with_clip
            for i in range(0, len(titles), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_titles = titles[i:i+batch_size]
                try:
                    vecs = embed_text_with_clip(batch_titles, batch_size=len(batch_titles))
                except Exception:
                    # if embedding fails, skip this batch
                    continue
                for pid, title, vec in zip(batch_ids, batch_titles, vecs):
                    arr = np.asarray(vec, dtype=float)
                    if arr.size != qvec.size:
                        continue
                    norm = np.linalg.norm(arr)
                    if norm == 0:
                        continue
                    arr = arr / norm
                    sim = float(np.dot(qvec, arr))
                    # find corresponding row metadata
                    # index i + local offset gives original row index
                    orig_idx = ids.index(pid)
                    r = all_rows[orig_idx]
                    sims.append((sim, r))

        sims.sort(reverse=True, key=lambda x: x[0])
        for sim, r in sims[:q.top_k]:
            out.append({'id': r.get('id'), 'title': r.get('title'), 'page_url': r.get('page_url'), 'image_urls': r.get('image_urls') or [], 'dist': float(sim)})
        return {'results': out}
    except Exception as e:
        # If DB or network failed, return an informative error and the image embedding (so client can decide)
        try:
            embed_vec = embed_image_url(q.image_url)
            return {'results': [], 'error': str(e), 'image_embedding': embed_vec.tolist()}
        except Exception:
            return {'results': [], 'error': str(e)}


def _fetch_image_bytes(url, timeout=8):
    """Fetch image bytes with retries and sensible timeouts. Returns bytes or raises."""
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


@app.post('/search/image-upload')
async def search_image_upload(file: UploadFile = File(...), top_k: int = 5):
    """Accept an uploaded image and return nearest-neighbor products by image embedding.

    This avoids network/URL issues by sending the image bytes directly.
    """
    data = await file.read()
    try:
        vec = embed_image_bytes(data)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': 'failed to embed image', 'detail': str(e)})

    # try to query DB using image embedding
    try:
        # reuse query_similar_by_image_url by temporarily writing a helper that accepts a vector is not available,
        # so we perform a DB-level vector query here similar to embeddings module
        from .embeddings import PGVector, _PGVECTOR_AVAILABLE, get_engine
        qvec = vec.tolist()
        engine = get_engine()
        with engine.connect() as conn:
            q = "SELECT id, title, page_url, image_urls, image_embedding <#> :v AS dist FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <#> :v LIMIT :k"
            if _PGVECTOR_AVAILABLE:
                try:
                    vparam = PGVector(qvec)
                    rows = conn.execute(q, {'v': vparam, 'k': top_k}).fetchall()
                except Exception:
                    vstr = '[' + ','.join(map(str, qvec)) + ']'
                    q2 = q.replace(':v', "'" + vstr + "'::vector")
                    rows = conn.execute(q2, {'k': top_k}).fetchall()
            else:
                rows = conn.execute(q, {'v': qvec, 'k': top_k}).fetchall()

        out = []
        for r in rows:
            if hasattr(r, '_mapping'):
                m = r._mapping
                id_ = m.get('id')
                title = m.get('title')
                page_url = m.get('page_url')
                image_urls = m.get('image_urls') or []
                dist = float(m.get('dist'))
            else:
                id_ = r[0]
                title = r[1]
                page_url = r[2]
                image_urls = r[3] if len(r) > 3 and r[3] is not None else []
                dist = float(r[-1])
            out.append({'id': id_, 'title': title, 'page_url': page_url, 'image_urls': image_urls, 'dist': dist})
        return {'results': out}
    except Exception as e:
        # DB failed; return the embedding so client can use local ANN or inspect
        return JSONResponse(status_code=200, content={'results': [], 'warning': 'db query failed', 'image_embedding': vec.tolist(), 'detail': str(e)})


