from playwright.sync_api import sync_playwright
from .utils import random_ua, polite_sleep
from bs4 import BeautifulSoup
import re
import logging

logger = logging.getLogger(__name__)


def _extract_generic(html, base_url=None):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title').get_text(strip=True) if soup.find('title') else None
    desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    description = desc_tag['content'] if desc_tag and desc_tag.get('content') else None
    images = []
    for img in soup.find_all('img')[:10]:
        src = img.get('data-src') or img.get('src')
        if src:
            images.append(src)
    price = None
    # naive price search
    text = soup.get_text(separator=' ')
    m = re.search(r'\$\s?[0-9\,]+(?:\.[0-9]{2})?', text)
    if m:
        price = float(re.sub(r'[\$,]', '', m.group(0)))
    return {
        'title': title,
        'description': description,
        'image_urls': images,
        'price': price,
        'currency': None,
    }


def _parse_amazon(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.select_one('#productTitle')
    title = title.get_text(strip=True) if title else None
    desc = None
    desc_tag = soup.select_one('#feature-bullets')
    if desc_tag:
        desc = ' '.join(li.get_text(strip=True) for li in desc_tag.select('li'))
    image_urls = []
    img_tag = soup.select_one('#landingImage')
    if img_tag and img_tag.get('data-old-hires'):
        image_urls.append(img_tag['data-old-hires'])
    # fallback generic
    gen = _extract_generic(html)
    for k, v in gen.items():
        if k == 'image_urls' and not image_urls:
            image_urls = v
        elif k == 'title' and not title:
            title = v
        elif k == 'description' and not desc:
            desc = v
    return {
        'title': title,
        'description': desc,
        'image_urls': image_urls,
        'price': None,
        'currency': None,
    }


def _parse_flipkart(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.select_one('span.B_NuCI')
    title = title.get_text(strip=True) if title else None
    desc = None
    desc_tag = soup.select_one('div._1rcHFq')
    if desc_tag:
        desc = desc_tag.get_text(strip=True)
    images = []
    for img in soup.select('img._396cs4'):
        src = img.get('src')
        if src:
            images.append(src)
    gen = _extract_generic(html)
    for k, v in gen.items():
        if k == 'image_urls' and not images:
            images = v
        elif k == 'title' and not title:
            title = v
        elif k == 'description' and not desc:
            desc = v
    return {
        'title': title,
        'description': desc,
        'image_urls': images,
        'price': None,
        'currency': None,
    }


SITE_HANDLERS = {
    'amazon.': _parse_amazon,
    'flipkart.': _parse_flipkart,
}


def parse_by_host(url, html):
    for host_key, fn in SITE_HANDLERS.items():
        if host_key in url:
            return fn(html)
    return _extract_generic(html)


def scrape_url(url, wait_for=2000, headless=True, timeout=30000):
    # returns dict with title, description, image_urls, price, currency
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(user_agent=random_ua())
        page = context.new_page()
        try:
            page.goto(url, timeout=timeout)
            page.wait_for_timeout(wait_for)
            html = page.content()
            data = parse_by_host(url, html)
            data['page_url'] = url
            # extract review snippets (best-effort)
            data['reviews'] = []
            try:
                if 'amazon.' in url:
                    data['reviews'] = extract_amazon_reviews(page, max_pages=3)
                else:
                    data['reviews'] = extract_reviews(html, url)
            except Exception:
                data['reviews'] = []
            return data
        finally:
            context.close()
            browser.close()


def extract_reviews(html, url=None, limit: int = 10):
    """Best-effort extraction of visible review snippets from a product page.
    Returns a list of dicts: {'review_text','rating','author'}.
    """
    soup = BeautifulSoup(html, 'html.parser')
    reviews = []

    # site-specific: Amazon
    if url and 'amazon.' in url:
        for div in soup.select('div[data-hook="review"]')[:limit]:
            text = div.select_one('span.review-text') or div.select_one('span[data-hook="review-body"]')
            rt = text.get_text(strip=True) if text else None
            author = div.select_one('span.a-profile-name')
            author = author.get_text(strip=True) if author else None
            reviews.append({'review_text': rt, 'rating': None, 'author': author})
        if reviews:
            return reviews

    # site-specific: Flipkart
    if url and 'flipkart.' in url:
        for div in soup.select('div._16PBlm, div._27M-vq')[:limit]:
            rt = div.get_text(separator=' ', strip=True)
            reviews.append({'review_text': rt, 'rating': None, 'author': None})
        if reviews:
            return reviews

    # Generic heuristic: look for elements with class or id containing 'review'
    candidates = []
    for el in soup.find_all(attrs={})[:0]:
        pass
    # gather nodes whose class or id contains 'review'
    for el in soup.find_all():
        cls = ' '.join(el.get('class') or [])
        eid = el.get('id') or ''
        name = el.name or ''
        if 'review' in cls.lower() or 'review' in eid.lower():
            text = el.get_text(separator=' ', strip=True)
            if text and len(text) > 30:
                candidates.append(text)
        # also consider paragraphs with 'review' nearby
        if len(candidates) >= limit:
            break

    for t in candidates[:limit]:
        reviews.append({'review_text': t, 'rating': None, 'author': None})

    # Fallback: grab sentences that look like review snippets ("I bought this", "works well")
    if not reviews:
        text = soup.get_text(separator='\n')
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for l in lines:
            if len(l) > 60 and (l.lower().startswith('i ') or 'works' in l.lower() or 'bought' in l.lower() or 'review' in l.lower()):
                reviews.append({'review_text': l, 'rating': None, 'author': None})
            if len(reviews) >= limit:
                break

    return reviews


def _clean_review_text(text: str) -> str:
    if not text:
        return ''
    # remove repeated whitespace
    t = re.sub(r'\s+', ' ', text).strip()
    # remove common Amazon artifacts
    t = re.sub(r'Customer reviews.*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\d+(?:\.\d+)? out of 5 stars', '', t, flags=re.IGNORECASE)
    t = t.strip(' -–:')
    return t


def extract_amazon_reviews(page, max_pages: int = 3, per_page_limit: int = 10):
    """Navigate Amazon 'see all reviews' and paginate to collect structured reviews.
    Returns list of {'review_text','rating','author'}.
    """
    reviews = []
    # try to find "see all reviews" link
    try:
        # common selector for see-all-reviews
        link = None
        for sel in ['a[data-hook="see-all-reviews-link-foot"]', 'a[data-hook="see-all-reviews-link"]', 'a[href*="/product-reviews/"]']:
            loc = page.query_selector(sel)
            if loc:
                link = loc.get_attribute('href')
                break
        if link:
            # navigate absolute/relative
            if link.startswith('http'):
                review_url = link
            else:
                base = page.url.split('?')[0]
                parsed = base.rsplit('/', 1)[0]
                review_url = page.url.split('/dp/')[0] + link if '/product-reviews/' in link else page.url + link
            page.goto(review_url)
            page.wait_for_timeout(1000)
    except Exception:
        # fall back to product page content
        pass

    page_count = 0
    while page_count < max_pages:
        page_count += 1
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        items = soup.select('div[data-hook="review"]')
        for it in items:
            body = it.select_one('span[data-hook="review-body"]') or it.select_one('span.review-text')
            txt = body.get_text(separator=' ', strip=True) if body else ''
            txt = _clean_review_text(txt)
            if not txt:
                continue
            author = it.select_one('span.a-profile-name')
            author = author.get_text(strip=True) if author else None
            rating = None
            rtag = it.select_one('i[data-hook="review-star-rating"]') or it.select_one('i[data-hook="cmps-review-star-rating"]')
            if rtag:
                m = re.search(r'(\d+(?:\.\d+)?)', rtag.get_text())
                if m:
                    try:
                        rating = float(m.group(1))
                    except Exception:
                        rating = None
            reviews.append({'review_text': txt, 'rating': rating, 'author': author})
            if len(reviews) >= max_pages * per_page_limit:
                break
        # try to go to next page
        if len(reviews) >= max_pages * per_page_limit:
            break
        try:
            nxt = page.query_selector('li.a-last a') or page.query_selector('a[data-hook="pagination-next"]')
            if nxt:
                nxt.click()
                page.wait_for_timeout(1200)
                continue
        except Exception:
            pass
        break

    # dedupe while preserving order
    seen = set()
    out = []
    for r in reviews:
        key = (r.get('author') or '') + '||' + (r.get('review_text') or '')
        key_norm = re.sub(r'\s+', ' ', key).strip().lower()
        if key_norm in seen:
            continue
        seen.add(key_norm)
        out.append(r)
    return out
