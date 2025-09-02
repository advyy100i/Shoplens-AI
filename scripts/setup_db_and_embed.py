import os
import sys
import traceback

# Configure DB connection to local compose mapping
os.environ['DATABASE_URL'] = os.getenv('DATABASE_URL', 'postgres://postgres:postgres@127.0.0.1:5432/shoplens')

try:
    from scraper.db import Base, get_engine
    from scraper.embeddings import embed_missing_image_embeddings, embed_missing_clip_text_embeddings
    from scraper.embeddings import ensure_vector_columns, ensure_clip_text_column
    engine = get_engine()
    print('Creating tables (if not exist)')
    Base.metadata.create_all(engine)
    print('Ensuring vector/clip_text columns exist')
    ensure_vector_columns()
    ensure_clip_text_column()
    print('Running limited image embedding fill (limit=200)')
    n = embed_missing_image_embeddings(limit=200, batch_size=16)
    print('image embeddings computed for', n, 'products')
    print('Running limited CLIP-text embedding fill (limit=200)')
    m = embed_missing_clip_text_embeddings(limit=200, batch_size=64)
    print('clip text embeddings computed for', m, 'products')
except Exception as e:
    print('Error during setup:', e)
    traceback.print_exc()
    sys.exit(2)
