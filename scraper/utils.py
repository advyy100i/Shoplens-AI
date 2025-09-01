import random
import time

USER_AGENTS = [
    # A small rotation set; extend as needed
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
]


def random_ua():
    return random.choice(USER_AGENTS)


def polite_sleep(min_s=1.0, max_s=3.0):
    time.sleep(random.uniform(min_s, max_s))
