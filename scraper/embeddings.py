import os
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForSequenceClassification, pipeline, AutoModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch
import asyncio
import numpy as np
import json
try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except Exception:
    _AIOHTTP_AVAILABLE = False

from .db import get_engine, get_session
from sqlalchemy import text
try:
    from pgvector import Vector as PGVector
    _PGVECTOR_AVAILABLE = True
except Exception:
    _PGVECTOR_AVAILABLE = False

# load models lazily
_text_model = None
_img_processor = None
_img_model = None
_device = None
_summarizer = None


def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _text_model


def get_image_model():
    global _img_processor, _img_model
    if _img_processor is None:
        # initialize CLIP processor and model and move model to the correct device
        _img_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        _img_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        # determine device once
        global _device
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            _img_model.to(_device)
        except Exception:
            # if moving to device fails, keep model on default and fall back to CPU at runtime
            _device = torch.device('cpu')
    return _img_processor, _img_model


def embed_text_with_clip(texts, batch_size: int = 32):
    """Embed texts using CLIP text encoder to share the same space as image embeddings."""
    proc, model = get_image_model()
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = proc(text=batch, return_tensors='pt', padding=True, truncation=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(_device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        vecs = out.cpu().numpy()
        # normalize
        for v in vecs:
            arr = np.asarray(v, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            all_vecs.append(arr)
    return np.stack(all_vecs)


def ensure_clip_text_column():
    engine = get_engine()
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        try:
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS clip_text_embedding vector(512);"))
        except Exception:
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS clip_text_embedding JSONB;"))
    finally:
        conn.close()


def store_product_clip_text_embedding(product_id, vector):
    engine = get_engine()
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    with engine.connect() as conn:
        conn.execute(text('UPDATE products SET clip_text_embedding = :v WHERE id = :id'), {'v': arr.tolist(), 'id': int(product_id)})
        conn.commit()


def embed_missing_clip_text_embeddings(limit: int = 500, batch_size: int = 64):
    """Compute CLIP text embeddings for product titles (if missing) and store them."""
    ensure_clip_text_column()
    engine = get_engine()
    sel = text("SELECT id, title FROM products WHERE (clip_text_embedding IS NULL) LIMIT :l")
    with engine.connect() as conn:
        rows = conn.execute(sel, { 'l': limit }).mappings().all()

    ids = [r.get('id') for r in rows]
    titles = [r.get('title') or '' for r in rows]
    if not ids:
        return 0

    vecs = embed_text_with_clip(titles, batch_size=batch_size)
    for pid, vec in zip(ids, vecs):
        store_product_clip_text_embedding(pid, vec)
    return len(ids)


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if torch.cuda.is_available() else -1)
    return _summarizer


def embed_texts(texts, batch_size: int = 32):
    """Embed a list of texts using the sentence-transformers model in batches.

    Args:
        texts: list[str] - input texts to embed
        batch_size: int - encoder batch size

    Returns:
        numpy.ndarray or list: embeddings in the same order as input texts
    """
    model = get_text_model()
    # sentence-transformers `encode` accepts a batch_size parameter which will
    # automatically process the inputs in batches. We disable the progress bar
    # by default for quieter runs.
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)


def embed_image_url(url):
    proc, model = get_image_model()
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert('RGB')
    inputs = proc(images=img, return_tensors='pt')
    # move tensors to model device if available
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(_device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        vec = out[0].cpu().numpy()
        return vec
    except Exception:
        # fallback: move inputs to CPU and run there
        try:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.cpu()
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            vec = out[0].cpu().numpy()
            return vec
        except Exception as e:
            raise


def embed_image_bytes(img_bytes):
    """Embed an image given raw bytes (sync)."""
    proc, model = get_image_model()
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    inputs = proc(images=img, return_tensors='pt')
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    vec = out[0].cpu().numpy()
    return vec


async def embed_image_url_async(url, timeout=10):
    """Asynchronously fetch image bytes and run embedding in a thread to avoid blocking the event loop.

    Uses aiohttp when available; otherwise falls back to running requests.get in a thread.
    """
    if _AIOHTTP_AVAILABLE:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.read()
        # run CPU-bound embedding in a thread
        vec = await asyncio.to_thread(embed_image_bytes, data)
        return vec
    else:
        # fallback: perform requests.get in a thread
        def _fetch():
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content

        data = await asyncio.to_thread(_fetch)
        vec = await asyncio.to_thread(embed_image_bytes, data)
        return vec


def summarize_reviews(review_texts, max_length=100):
    summarizer = get_summarizer()
    joined = '\n'.join(review_texts)
    # split if too long
    if len(joined) < 1024:
        s = summarizer(joined, max_length=max_length, min_length=30, do_sample=False)
        return s[0]['summary_text']
    # fallback: summarize first N reviews
    s = summarizer('\n'.join(review_texts[:10]), max_length=max_length, min_length=30, do_sample=False)
    return s[0]['summary_text']


def ensure_vector_columns():
    engine = get_engine()
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        # Check if 'vector' type exists in the database (pgvector installed)
        try:
            has_vector = conn.execute(text("SELECT 1 FROM pg_type WHERE typname = 'vector';")).fetchone() is not None
        except Exception:
            has_vector = False

        if has_vector:
            # create vector columns (will succeed if type exists)
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS product_embedding vector(384);"))
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS image_embedding vector(512);"))
        else:
            # create JSONB fallback columns
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS product_embedding JSONB;"))
            conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS image_embedding JSONB;"))
    finally:
        conn.close()


def ensure_status_columns():
    """Add per-product embedding status/error/timestamp columns if missing."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS embedding_text_status VARCHAR(32);
        """))
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS embedding_image_status VARCHAR(32);
        """))
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS embedding_text_error TEXT;
        """))
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS embedding_image_error TEXT;
        """))
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS embedding_text_updated_at TIMESTAMP WITH TIME ZONE;
        """))
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS embedding_image_updated_at TIMESTAMP WITH TIME ZONE;
        """))
        conn.commit()


def store_product_text_embedding(product_id, vector):
    engine = get_engine()
    # vector must be list/ndarray; normalize to unit length for cosine similarity
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    with engine.connect() as conn:
        # Try updating vector column; if not available, update JSONB
        try:
            conn.execute(text('UPDATE products SET product_embedding = :v WHERE id = :id'), {'v': arr.tolist(), 'id': int(product_id)})
        except Exception:
            conn.execute(text('UPDATE products SET product_embedding = :v WHERE id = :id'), {'v': json.dumps(arr.tolist()), 'id': int(product_id)})
        conn.commit()


def store_product_image_embedding(product_id, vector):
    engine = get_engine()
    # normalize vector to unit length
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    with engine.connect() as conn:
        try:
            conn.execute(text('UPDATE products SET image_embedding = :v WHERE id = :id'), {'v': arr.tolist(), 'id': int(product_id)})
        except Exception:
            conn.execute(text('UPDATE products SET image_embedding = :v WHERE id = :id'), {'v': json.dumps(arr.tolist()), 'id': int(product_id)})
        conn.commit()


def query_similar_by_text(query, top_k=5):
    vec = embed_texts([query])[0]
    # normalize query vector
    v = np.asarray(vec, dtype=float)
    vnorm = np.linalg.norm(v)
    if vnorm > 0:
        v = (v / vnorm).tolist()
    else:
        v = v.tolist()
    engine = get_engine()
    with engine.connect() as conn:
        # requires pgvector; ensure the parameter is adapted to pgvector.Vector
        q = text('SELECT id, title, page_url, image_urls, product_embedding <#> :v AS dist FROM products WHERE product_embedding IS NOT NULL ORDER BY product_embedding <#> :v LIMIT :k')
        if _PGVECTOR_AVAILABLE:
            try:
                vparam = PGVector(v)
                rows = conn.execute(q, {'v': vparam, 'k': top_k}).fetchall()
                return rows
            except Exception:
                # fallback to Python-side NN below
                pass

        # If pgvector unavailable or SQL-level query failed, perform Python-side nearest neighbors.
        rows = conn.execute(text('SELECT id, title, page_url, image_urls, product_embedding FROM products WHERE product_embedding IS NOT NULL')).mappings().all()

    # compute cosine similarities in Python
    sims = []
    for r in rows:
        emb = r.get('product_embedding')
        if not emb:
            continue
        try:
            arr = np.asarray(json.loads(emb) if isinstance(emb, str) else emb, dtype=float)
        except Exception:
            try:
                import ast as _ast
                arr = np.asarray(_ast.literal_eval(emb), dtype=float)
            except Exception:
                continue
        if arr.size != len(v):
            continue
        norm = np.linalg.norm(arr)
        if norm == 0:
            continue
        arr = arr / norm
        sim = float(np.dot(np.asarray(v, dtype=float), arr))
        sims.append((sim, r))

    sims.sort(reverse=True, key=lambda x: x[0])
    # return list of SQLAlchemy Row-like mappings to match existing callers
    return [ r for sim, r in sims[:top_k] ]


def query_similar_by_image_url(image_url, top_k=5):
    vec = embed_image_url(image_url)
    # normalize query vector
    v = np.asarray(vec, dtype=float)
    vnorm = np.linalg.norm(v)
    if vnorm > 0:
        v = (v / vnorm).tolist()
    else:
        v = v.tolist()
    engine = get_engine()
    with engine.connect() as conn:
        q = text('SELECT id, title, page_url, image_urls, image_embedding <#> :v AS dist FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <#> :v LIMIT :k')
        if _PGVECTOR_AVAILABLE:
            try:
                vparam = PGVector(v)
                rows = conn.execute(q, {'v': vparam, 'k': top_k}).fetchall()
                return rows
            except Exception:
                # fall through to Python-side NN
                pass

        # fallback: fetch all image embeddings and compute cosine similarity in Python
        rows = conn.execute(text('SELECT id, title, page_url, image_urls, image_embedding FROM products WHERE image_embedding IS NOT NULL')).mappings().all()

    sims = []
    for r in rows:
        emb = r.get('image_embedding')
        if not emb:
            continue
        try:
            arr = np.asarray(json.loads(emb) if isinstance(emb, str) else emb, dtype=float)
        except Exception:
            try:
                import ast as _ast
                arr = np.asarray(_ast.literal_eval(emb), dtype=float)
            except Exception:
                continue
        if arr.size != v.size:
            continue
        norm = np.linalg.norm(arr)
        if norm == 0:
            continue
        arr = arr / norm
        sim = float(np.dot(v, arr))
        sims.append((sim, r))

    sims.sort(reverse=True, key=lambda x: x[0])
    return [ r for sim, r in sims[:top_k] ]


def embed_missing_image_embeddings(limit: int = 100, batch_size: int = 8):
    """Find products with image_urls but missing image_embedding and compute embeddings for them.

    This is a small helper to populate image embeddings on-demand for a limited number of products.
    It fetches the first image URL if available and stores the resulting vector.
    """
    engine = get_engine()
    sel = text("SELECT id, image_urls FROM products WHERE (image_embedding IS NULL) AND (image_urls IS NOT NULL) LIMIT :l")
    with engine.connect() as conn:
        rows = conn.execute(sel, { 'l': limit }).mappings().all()

    ids_and_urls = []
    for r in rows:
        img_list = r.get('image_urls') or []
        if img_list:
            # image_urls stored as list; take first
            first = img_list[0]
            ids_and_urls.append((r.get('id'), first))

    # process in small batches
    for i in range(0, len(ids_and_urls), batch_size):
        batch = ids_and_urls[i:i+batch_size]
        for pid, url in batch:
            try:
                vec = embed_image_url(url)
                store_product_image_embedding(pid, vec)
                # set status columns if present
                try:
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE products SET embedding_image_status = 'done', embedding_image_error = NULL, embedding_image_updated_at = now() WHERE id = :id"), { 'id': pid })
                        conn.commit()
                except Exception:
                    pass
            except Exception as e:
                # record error
                try:
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE products SET embedding_image_status = 'error', embedding_image_error = :err WHERE id = :id"), { 'id': pid, 'err': str(e) })
                        conn.commit()
                except Exception:
                    pass

    return len(ids_and_urls)
