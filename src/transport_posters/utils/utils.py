from typing import Optional
import logging
from unidecode import unidecode
import re

logger = logging.getLogger(__name__)


def natural_key(route_id: str) -> tuple:
    """Return a tuple key for natural sorting of mixed alphanumeric strings."""
    parts = re.findall(r'\d+|\D+', route_id)
    key = []
    for p in parts:
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    return tuple(key)


def slugify(text: Optional[str]) -> str:
    """Convert text into a URL-friendly slug using ASCII characters only."""
    text = str(text or "")
    ascii_ = unidecode(text)
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")
    return slug[:40] or "unnamed"
