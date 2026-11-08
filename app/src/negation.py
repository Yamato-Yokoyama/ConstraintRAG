"""Detect negated queries and check products for the excluded term."""
import re

NEGATION = re.compile(r"\b(?:without|not|no|non|free|except|exclu\w*)\b", re.IGNORECASE)
EXCLUDED_AFTER = re.compile(r"\b(?:without|not|no|non|except|excluding)\s+(.+)$", re.IGNORECASE)
EXCLUDED_BEFORE = re.compile(r"([\w'-]+)[\s-]+free\b", re.IGNORECASE)
NEGATOR_BEFORE = re.compile(
    r"\b(?:without|no|not|non|except|excluding|free\s+of)\b[\s-]+"
    r"(?:(?!(?:with|and|or|plus|but|includes?|including)\b)[\w'-]+\s+){0,2}$",
    re.IGNORECASE,
)
FREE_AFTER = re.compile(r"^[\s-]*free\b", re.IGNORECASE)
TRAP_SYNONYMS = {"basket": ["basket", "wastebasket"]}


def is_negated(query: str) -> bool:
    return bool(NEGATION.search(query))


def extract_excluded_term(query: str) -> str | None:
    """Return the text after the negator. It can include extra words, such as "hood mens"."""
    match = EXCLUDED_AFTER.search(query)
    if match:
        return match.group(1).strip()
    match = EXCLUDED_BEFORE.search(query)
    return match.group(1).strip() if match else None


def has_unnegated(text: str, term: str) -> bool:
    """True if term appears at least once without being negated in text."""
    words = TRAP_SYNONYMS.get(term.lower(), [term])
    alternatives = "|".join(re.escape(word) for word in words)
    pattern = re.compile(r"\b(?:" + alternatives + r")\w*", re.IGNORECASE)
    for match in pattern.finditer(text):
        negated = NEGATOR_BEFORE.search(text[: match.start()])
        free = FREE_AFTER.search(text[match.end() :])
        if not (negated or free):
            return True
    return False
