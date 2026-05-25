import logging
import sys
import functools
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    stream=sys.stdout
)

logger = logging.getLogger("wc2026")

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def _safe_repr(arg):
            s = str(arg)
            return s if len(s) <= 30 else f"{s[:47]}..."
        args_str = ", ".join(map(_safe_repr, args))
        kwargs_str = ", ".join(f"{k}={_safe_repr(v)}" for k, v in kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        start_time = time.time()
        logger.info(f"Запуск {func.__name__}({all_args})...")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Выполнена {func.__name__}({all_args}) -> {end_time - start_time:.4f} сек.")
        return result
    return wrapper


ELORATINGS_URL_1 = "https://eloratings.net/2026_World_Cup_qualifying_results"
ELORATINGS_URL_2 = "https://eloratings.net/2026_World_Cup_qualifying_playoffs_results"

WC2026_QUALIFIERS_FILE = "wc2026_qualifiers"
BASIC_FEATURES_FILE = "wc2026_basic_features"

PREV_N = 5

FEATURES_LIST = ["home_elo_rating", "away_elo_rating", "home_elo_rank", "away_elo_rank", "home_market_value", "away_market_value", "prev_N_pts_H", "prev_N_pts_A"]

