"""
Utility to shorten street/stop names by abbreviating common road type words,
with automatic language detection.

Dependencies:
    pip install langdetect
"""

import re
import logging
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

_ABBREVIATIONS = {
    "ru": {
        "улица": "ул",
        "площадь": "пл",
        "проспект": "пр-т",
        "бульвар": "бул",
        "шоссе": "ш",
        "переулок": "пер",
        "микрорайон": "мкр",
        "проезд": "пр-д",
        "тупик": "туп",
        "набережная": "наб",
        "аллея": "ал",

        "техникум": "техн",
        "музыкальное": "муз",
        "музыкальный": "муз",
        "областная": "обл",
        "железнодорожный": "ж/д",
        "автомобильный": "авт",
        "авиационный": "авиац",
        "морской": "мор",
        "строительный": "стр"
    }
    ,
    "en": {
        "street": "St.",
        "avenue": "Ave.",
        "road": "Rd.",
        "boulevard": "Blvd.",
        "square": "Sq.",
        "drive": "Dr.",
        "parkway": "Pkwy.",
    },
    "es": {
        "calle": "c.",
        "avenida": "av.",
        "plaza": "pl.",
        "paseo": "p.º",
        "carretera": "ctra.",
    },
}

_PUNCT_TO_SPACE = re.compile(r"[;,]+")
_MULTISPACE = re.compile(r"\s+")


def _apply_abbreviations(text: str, lang: str) -> str:
    """Replace full words with their abbreviations for the given language."""
    rules = _ABBREVIATIONS[lang]
    for full, abbr in rules.items():
        pattern = rf"\b{re.escape(full)}\b\.?"
        text = re.sub(pattern, abbr, text, flags=re.IGNORECASE)
    return text


def shorten_stop_name(name: str) -> str:
    """
    Shorten a stop/street name.

    If the language can be detected and is supported, common place‑type words
    are replaced with their abbreviations. Excess punctuation and duplicate
    whitespace are removed.

    If language detection fails or the language has no rules, the original
    name is returned unchanged.

    Args:
        name: Full stop/street name.

    Returns:
        Shortened, cleaned name string.
    """
    original = name

    _SUPPORTED_LANGS = [
        Language.RUSSIAN,
        Language.ENGLISH,
        Language.SPANISH,
    ]
    _detector = LanguageDetectorBuilder.from_languages(*_SUPPORTED_LANGS).build()

    lang_obj = _detector.detect_language_of(name)

    if lang_obj is None:
        logger.warning("Unable to detect language for stop name: %r", name)
        return original
    lang = lang_obj.iso_code_639_1.name.lower()

    if lang not in _ABBREVIATIONS:
        logger.warning("Unsupported language '%s' for stop name: %r", lang, name)
        return original

    result = _apply_abbreviations(name, lang)
    result = _PUNCT_TO_SPACE.sub(" ", result)
    result = _MULTISPACE.sub(" ", result).strip()

    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result


if __name__ == "__main__":
    samples = [
        "Проспект Мира",
        "улица Ленина, дом 10",
        "Times Square Boulevard",
        "calle de Alcalá",
        "Unknown-lang Straße",
    ]
    for s in samples:
        print(f"{s} -> {shorten_stop_name(s)}")
