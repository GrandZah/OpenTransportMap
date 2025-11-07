import logging
from datetime import datetime
from functools import wraps
from pathlib import Path
import time

from transport_posters.load_configs import CONFIG_PATHS

logger = logging.getLogger(__name__)


def configure_root_logger():
    """Configure the root logger to write detailed debug logs to a timestamped file
    and concise info logs to the console, while silencing noisy third-party loggers."""

    log_dir = Path(CONFIG_PATHS.output_logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = log_dir / f"app_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    root.addHandler(ch)

    for noisy in ('matplotlib.font_manager', 'matplotlib._png', 'PIL.PngImagePlugin'):
        logging.getLogger(noisy).disabled = True


def log_function_call(func=None, *, enable_timing=False):
    """Decorator with optional timing and default behavior"""

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time() if enable_timing else None
            logger.info(f"START: {f.__name__}")

            try:
                result = f(*args, **kwargs)
                msg = f"SUCCESS: {f.__name__}"
                if enable_timing:
                    msg += f" completed in {time.time() - start_time:.4f}s"
                logger.info(msg)
                return result

            except Exception as e:
                msg = f"ERROR: {f.__name__} failed"
                if enable_timing:
                    msg += f" after {time.time() - start_time:.4f}s"
                logger.error(f"{msg}: '{str(e)}'")
                raise

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
