from .db import get_session, Source, Product, Review
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select


def get_or_create_source(session, name, base_url):
    stmt = select(Source).where(Source.base_url == base_url)
    r = session.execute(stmt).scalars().first()
    if r:
        return r
    src = Source(name=name, base_url=base_url)
    session.add(src)
    session.commit()
    return src


def save_product(session, source, product_data: dict):
    # product_data: source_product_id, title, description, price, currency, image_urls, page_url
    stmt = select(Product).where(Product.source_id == source.id, Product.source_product_id == product_data.get('source_product_id'))
    existing = session.execute(stmt).scalars().first()
    if existing:
        return existing
    p = Product(
        source_id=source.id,
        source_product_id=product_data.get('source_product_id'),
        title=product_data.get('title'),
        description=product_data.get('description'),
        price=product_data.get('price'),
        currency=product_data.get('currency'),
        image_urls=product_data.get('image_urls') or [],
        page_url=product_data.get('page_url')
    )
    session.add(p)
    try:
        session.commit()
        return p
    except IntegrityError:
        session.rollback()
        return session.execute(stmt).scalars().first()


def save_review(session, product, review_data: dict):
    # review_data: review_text, rating, author
    text = (review_data.get('review_text') or '').strip()
    author = (review_data.get('author') or '').strip()
    # simple dedupe: exact match on product and review_text or author+text
    stmt = select(Review).where(Review.product_id == product.id)
    for existing in session.execute(stmt).scalars().all():
        et = (existing.review_text or '').strip()
        ea = (existing.author or '').strip()
        if et and text and et == text:
            return existing
        if ea and author and ea == author and et and text and et == text:
            return existing

    r = Review(
        product_id=product.id,
        review_text=text,
        rating=review_data.get('rating'),
        author=author or None
    )
    session.add(r)
    session.commit()
    return r
