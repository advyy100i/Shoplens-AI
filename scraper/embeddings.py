import os
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForSequenceClassification, pipeline, AutoModel
from PIL import Image
import requests
from io import BytesIO
import torch

from .db import get_engine, get_session
from sqlalchemy import text

# load models lazily
_text_model = None
_img_processor = None
_img_model = None
_summarizer = None


def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _text_model


def get_image_model():
    global _img_processor, _img_model
    if _img_processor is None:
        _img_processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
        _img_model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    return _img_processor, _img_model


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device=0 if torch.cuda.is_available() else -1)
    return _summarizer


def embed_texts(texts):
    model = get_text_model()
    return model.encode(texts, show_progress_bar=False)


def embed_image_url(url):
    proc, model = get_image_model()
    resp = requests.get(url, timeout=10)
    img = Image.open(BytesIO(resp.content)).convert('RGB')
    inputs = proc(images=img, return_tensors='pt')
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    vec = out[0].cpu().numpy()
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
    # Add columns product_embedding and image_embedding as vector if not exists
    with engine.connect() as conn:
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS product_embedding vector(384);
        """))
        conn.execute(text("""
        ALTER TABLE products
        ADD COLUMN IF NOT EXISTS image_embedding vector(512);
        """))
        conn.commit()


def store_product_text_embedding(product_id, vector):
    engine = get_engine()
    # vector must be list/ndarray
    with engine.connect() as conn:
        conn.execute(text('UPDATE products SET product_embedding = :v WHERE id = :id'), {'v': list(map(float, vector)), 'id': int(product_id)})
        conn.commit()


def store_product_image_embedding(product_id, vector):
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text('UPDATE products SET image_embedding = :v WHERE id = :id'), {'v': list(map(float, vector)), 'id': int(product_id)})
        conn.commit()


def query_similar_by_text(query, top_k=5):
    vec = embed_texts([query])[0].tolist()
    engine = get_engine()
    with engine.connect() as conn:
        # requires pgvector
        q = text('SELECT id, title, page_url, product_embedding <#> :v AS dist FROM products WHERE product_embedding IS NOT NULL ORDER BY product_embedding <#> :v LIMIT :k')
        rows = conn.execute(q, {'v': vec, 'k': top_k}).fetchall()
    return rows


def query_similar_by_image_url(image_url, top_k=5):
    vec = embed_image_url(image_url).tolist()
    engine = get_engine()
    with engine.connect() as conn:
        q = text('SELECT id, title, page_url, image_embedding <#> :v AS dist FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <#> :v LIMIT :k')
        rows = conn.execute(q, {'v': vec, 'k': top_k}).fetchall()
    return rows
