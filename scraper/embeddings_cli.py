import argparse
from .embeddings import ensure_vector_columns, embed_texts, embed_image_url, store_product_text_embedding, store_product_image_embedding
try:
    from .embeddings import embed_image_url_async
    _ASYNC_IMAGES = True
except Exception:
    _ASYNC_IMAGES = False
from .db import get_engine
from tqdm import tqdm
from sqlalchemy import text


def run_generate(all_products=False, text_only=False, image_only=False, limit=100, force=False, retry_failed=False, batch_size=32):
    """Process only rows that are missing embeddings and show per-item progress.

    This uses a direct SQL select so it can inspect the `product_embedding` and
    `image_embedding` columns which are added to the DB but not present on the
    SQLAlchemy ORM `Product` model.
    """
    ensure_vector_columns()
    # ensure status/error columns exist
    try:
        from .embeddings import ensure_status_columns
        ensure_status_columns()
    except Exception:
        pass
    engine = get_engine()

    # Build WHERE clause to select rows needing embeddings. Respect --force and --retry-failed
    where_clauses = []
    if not all_products and not force:
        if text_only:
            clause = 'product_embedding IS NULL'
            if retry_failed:
                clause = '(' + clause + " OR embedding_text_status = 'failed')"
            where_clauses.append(clause)
        elif image_only:
            clause = 'image_embedding IS NULL'
            if retry_failed:
                clause = '(' + clause + " OR embedding_image_status = 'failed')"
            where_clauses.append(clause)
        else:
            clause = '(product_embedding IS NULL OR image_embedding IS NULL)'
            if retry_failed:
                clause = '(' + clause + " OR embedding_text_status = 'failed' OR embedding_image_status = 'failed')"
            where_clauses.append(clause)

    sql = 'SELECT id, title, description, image_urls, product_embedding, image_embedding FROM products'
    if where_clauses:
        sql += ' WHERE ' + ' AND '.join(where_clauses)
    if not all_products:
        sql += f' LIMIT {int(limit)}'

    with engine.connect() as conn:
        # use mappings() so results are dict-like and support row['id'] access
        rows = conn.execute(text(sql)).mappings().all()

    if not rows:
        print('No products need embeddings (or no products found).')
        return

    # Option 1: synchronous loop (default)
    if not _ASYNC_IMAGES:
        for row in tqdm(rows, desc='embedding products', unit='item'):
            pid = row['id']
            title = row['title']
            desc = row['description']
            image_urls = row['image_urls'] or []
            has_text_vec = row['product_embedding'] is not None
            has_img_vec = row['image_embedding'] is not None

            # Text embedding
            if not image_only and (force or (not has_text_vec)) and (title or desc):
                try:
                    txt = (title or '') + '\n' + (desc or '')
                    v = embed_texts([txt], batch_size=batch_size)[0]
                    store_product_text_embedding(pid, v)
                    # mark success
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE products SET embedding_text_status='ok', embedding_text_error=NULL, embedding_text_updated_at=now() WHERE id = :id"), {'id': pid})
                    tqdm.write(f'[{pid}] text embedding saved')
                except Exception as e:
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE products SET embedding_text_status='failed', embedding_text_error=:err, embedding_text_updated_at=now() WHERE id = :id"), {'err': str(e), 'id': pid})
                    tqdm.write(f'[{pid}] text embedding FAILED: {e}')

            # Image embedding (first image)
            if not text_only and image_urls and (force or (not has_img_vec)):
                try:
                    v = embed_image_url(image_urls[0])
                    store_product_image_embedding(pid, v)
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE products SET embedding_image_status='ok', embedding_image_error=NULL, embedding_image_updated_at=now() WHERE id = :id"), {'id': pid})
                    tqdm.write(f'[{pid}] image embedding saved')
                except Exception as e:
                    with engine.connect() as conn:
                        conn.execute(text("UPDATE products SET embedding_image_status='failed', embedding_image_error=:err, embedding_image_updated_at=now() WHERE id = :id"), {'err': str(e), 'id': pid})
                    tqdm.write(f'[{pid}] image embedding FAILED: {e}')
    else:
        # Option 2: async image embedding with concurrency
        import asyncio
        from functools import partial

        async def _run_async():
            sem = asyncio.Semaphore(run_generate._concurrency)

            async def _process(row):
                pid = row['id']
                title = row['title']
                desc = row['description']
                image_urls = row['image_urls'] or []
                has_text_vec = row['product_embedding'] is not None
                has_img_vec = row['image_embedding'] is not None

                # Text embedding (still sync/batched)
                if not image_only and (force or (not has_text_vec)) and (title or desc):
                    try:
                        txt = (title or '') + '\n' + (desc or '')
                        v = embed_texts([txt], batch_size=batch_size)[0]
                        store_product_text_embedding(pid, v)
                        with engine.connect() as conn:
                            conn.execute(text("UPDATE products SET embedding_text_status='ok', embedding_text_error=NULL, embedding_text_updated_at=now() WHERE id = :id"), {'id': pid})
                        tqdm.write(f'[{pid}] text embedding saved')
                    except Exception as e:
                        with engine.connect() as conn:
                            conn.execute(text("UPDATE products SET embedding_text_status='failed', embedding_text_error=:err, embedding_text_updated_at=now() WHERE id = :id"), {'err': str(e), 'id': pid})
                        tqdm.write(f'[{pid}] text embedding FAILED: {e}')

                # Image embedding
                if not text_only and image_urls and (force or (not has_img_vec)):
                    async with sem:
                        try:
                            v = await embed_image_url_async(image_urls[0])
                            # store in thread to avoid blocking
                            await asyncio.to_thread(store_product_image_embedding, pid, v)
                            with engine.connect() as conn:
                                conn.execute(text("UPDATE products SET embedding_image_status='ok', embedding_image_error=NULL, embedding_image_updated_at=now() WHERE id = :id"), {'id': pid})
                            tqdm.write(f'[{pid}] image embedding saved')
                        except Exception as e:
                            with engine.connect() as conn:
                                conn.execute(text("UPDATE products SET embedding_image_status='failed', embedding_image_error=:err, embedding_image_updated_at=now() WHERE id = :id"), {'err': str(e), 'id': pid})
                            tqdm.write(f'[{pid}] image embedding FAILED: {e}')

            # schedule tasks
            tasks = [asyncio.create_task(_process(r)) for r in rows]
            for f in asyncio.as_completed(tasks):
                await f

        # attach a concurrency param to function for easy use
        if not hasattr(run_generate, '_concurrency'):
            run_generate._concurrency = 4
    asyncio.run(_run_async())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--image-only', action='store_true')
    parser.add_argument('--force', action='store_true', help='Recompute embeddings even if they exist')
    parser.add_argument('--retry-failed', action='store_true', help='Retry items previously marked as failed')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for text encoder')
    parser.add_argument('--concurrency', type=int, default=4, help='Max concurrent image downloads when aiohttp is available')
    args = parser.parse_args()
    # set concurrency on the function so the async worker can access it
    run_generate._concurrency = max(1, int(args.concurrency))
    run_generate(all_products=args.all, text_only=args.text_only, image_only=args.image_only, limit=args.limit, force=args.force, retry_failed=args.retry_failed, batch_size=args.batch_size)
