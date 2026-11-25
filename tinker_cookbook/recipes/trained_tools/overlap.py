"""Keep generated requests clear of a held-out test: no shared content 3-gram, no near-duplicate."""

import re

from tinker_cookbook.recipes.trained_tools import phrasings as P

# fmt: off
POOLS = (P.TOPICS, P.SCENES, P.QUESTIONS, P.URLS, P.FACTS, P.RECALL_KEYS, P.WORDS, P.TRANSLATE, P.REWORD,
         P.SMALLTALK, P.OPINIONS, P.EXPLAIN, P.BRAIN, P.FACTUAL, P.GAMEWORDS, P.LANGS, P.STYLES, P.CONCEPTS)  # fmt: skip
SLOT = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(v) for v in sorted({v for p in POOLS for v in p}, key=len, reverse=True))
    + r")(?!\w)"
)
_STOP_WORDS = "a an the me my you your it its i im we our us to of for on in at by with from into as is are was were be am do does did can could would should will and or but so if then than that this these those there here what which who when where why how not no yes any some one all more other such only too very just also give tell get show want need have make let please thanks hey hi okay about after before through between without arg"
STOP = frozenset(_STOP_WORDS.split())


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))


def grams(tokens: list[str]) -> set[tuple[str, ...]]:
    """Word 3-grams with at least one content word; function words alone are not a phrasing."""
    return {
        g
        for g in zip(tokens, tokens[1:], tokens[2:], strict=False)
        if any(t not in STOP for t in g)
    }


def frame(request: str) -> list[str]:
    """Words of a request with pooled slot values, URLs and numbers replaced by one ``arg`` token."""
    text = re.sub(r"https?://\S+|\d[\d\s*+\-/().,]*", " arg ", SLOT.sub(" arg ", request))
    tokens = words(text)
    return [t for i, t in enumerate(tokens) if t != "arg" or tokens[i - 1 : i] != ["arg"]]


class Blocker:
    """Rejects a request that shares a content 3-gram with, or is a near-duplicate of, a held-out one."""

    def __init__(self, held_out: list[str]):
        self.grams = set().union(*(grams(frame(t)) | grams(words(t)) for t in held_out))
        self.sets = [set(words(t)) for t in held_out]

    def blocked(self, request: str) -> bool:
        tokens = set(words(request))
        near = any(len(tokens & s) / max(1, len(tokens | s)) >= 0.6 for s in self.sets)
        return near or bool(grams(frame(request)) & self.grams)
