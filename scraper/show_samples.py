from .db import get_session, Source, Product, Review, get_engine


def show_samples(limit=5):
    engine = get_engine()
    session = get_session(engine)
    print('\nSources:')
    for s in session.query(Source).limit(limit).all():
        print(f'  id={s.id} name={s.name} base_url={s.base_url}')

    print('\nProducts:')
    for p in session.query(Product).limit(limit).all():
        print(f'  id={p.id} title={(p.title or '')[:80]!r} price={p.price} page_url={p.page_url}')

    print('\nReviews:')
    for r in session.query(Review).limit(limit).all():
        print(f'  id={r.id} product_id={r.product_id} author={r.author} text={(r.review_text or '')[:120]!r}')


if __name__ == '__main__':
    show_samples()
