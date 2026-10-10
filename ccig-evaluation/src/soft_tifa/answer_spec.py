"""
Parse a verbalize.py `subqa` expected-answer string into a typed spec, and score it
against a VQA model's per-candidate probabilities.

Every answer string ccig-dataset-gen's verbalize.py produces falls into exactly one
of four shapes (verified against the full C1-C9 verbalizer -- see
ccig-dataset-gen/src/common/verbalize.py):

  "yes" / "no"                     -> YesNoSpec
  "True" / "False"                 -> TrueFalseSpec
  "<op> <n>"        e.g. "> 0"     -> CountSpec, no symbol bound or referenced
  "<sym> <op> <n>"  e.g. "c > 0"   -> CountSpec, binds symbol <sym> to this
                                       question's own answer while also checking it
  "<op> <sym>"      e.g. "= c"     -> CountSpec, checks this question's answer
                                       against a symbol bound by an earlier question
  "<sym>"           e.g. "n"       -> CountSpec, binds <sym> only -- no check (an
                                       "anchor" question with no expected answer of
                                       its own; see excluded_from_score in scoring.py)
  anything else     e.g. "blue"    -> OpenValueSpec (a literal property value)

All "<sym> ..." forms only ever appear on questions phrased "How many ...?" -- the
symbol always names *this* question's own count for later questions in the same
subqa dict to refer back to (see scoring.py for how that reference is resolved).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_YES_NO = {"yes", "no"}
_TRUE_FALSE = {"True", "False"}

# optional single-letter symbol, comparator, then either an integer or another
# single-letter symbol. Leading/trailing whitespace (verbalize.py isn't fully
# consistent about it) is tolerated.
_COUNT_RE = re.compile(
    r"^\s*(?:(?P<lhs>[a-z])\s*)?(?P<op>>=|<=|=|>|<)\s*(?P<rhs>\d+|[a-z])\s*$"
)
_BARE_SYMBOL_RE = re.compile(r"^\s*(?P<sym>[a-z])\s*$")


@dataclass(frozen=True)
class YesNoSpec:
    expected: str  # "yes" | "no"


@dataclass(frozen=True)
class TrueFalseSpec:
    expected: str  # "True" | "False"


@dataclass(frozen=True)
class CountSpec:
    binds_symbol: str | None  # e.g. "c" -- store this question's count under this name
    op: str | None  # None means "anchor only, nothing to check"
    rhs: int | str | None  # literal count, or a symbol name to resolve against, or None


@dataclass(frozen=True)
class OpenValueSpec:
    expected: str  # a literal property value, e.g. "blue", "cube", "r0"


AnswerSpec = YesNoSpec | TrueFalseSpec | CountSpec | OpenValueSpec


def parse_answer(raw: str) -> AnswerSpec:
    s = raw.strip()
    if s.lower() in _YES_NO:
        return YesNoSpec(expected=s.lower())
    if s in _TRUE_FALSE:
        return TrueFalseSpec(expected=s)

    m = _COUNT_RE.match(s)
    if m:
        rhs_raw = m.group("rhs")
        rhs: int | str = int(rhs_raw) if rhs_raw.isdigit() else rhs_raw
        lhs = m.group("lhs")
        if lhs is None:
            lhs = "a"
        return CountSpec(binds_symbol=lhs, op=m.group("op"), rhs=rhs)

    m = _BARE_SYMBOL_RE.match(s)
    if m:
        return CountSpec(binds_symbol=m.group("sym"), op=None, rhs=None)

    return OpenValueSpec(expected=s)


def compare(count: int, op: str, rhs: int) -> bool:
    if op == ">":
        return count > rhs
    if op == "<":
        return count < rhs
    if op == ">=":
        return count >= rhs
    if op == "<=":
        return count <= rhs
    if op == "=":
        return count == rhs
    raise ValueError(f"Unknown comparison operator: {op!r}")
