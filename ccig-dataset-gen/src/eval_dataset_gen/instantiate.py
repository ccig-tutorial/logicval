"""
Parse ASP constraint template files and generate valid placeholder assignments.

Placeholder naming convention (from template .txt files):
  P1', P2', P3', P4'  — property names (color/shape/size/region)
  V1', V2', V3', V4'  — property values (domain of their paired P)
  D1', D2', D3'       — spatial directions (left/right/front/behind)
  N'                  — integer count (1/2/3)

V_i is always paired with P_i (V1'←P1', V2'←P2', etc.).
Inequalities in the rule text (e.g. P1'!=P2') are respected automatically.
"""

import re
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set

from .domain import load_domain, COUNTS


# ── Regex patterns ─────────────────────────────────────────────────────────

# Matches any primed placeholder: P1', V2', D3', N'
# Uses (?<!\w) so we don't match mid-word; ' is not a word char so no trailing boundary needed.
_N_RE = re.compile(r"(?<!\w)(N')")

# Matches inequality constraints between primed placeholders: P1'!=P2', V1'!=V3', …
_INEQ_RE = re.compile(r"(?<!\w)([PVD][1-4]'|N')\s*!=\s*([PVD][1-4]'|N')")

_REMOVE_INEQ_RE = re.compile(r"\s*[PVD][1-4]'\s*!=\s*[PVD][1-4]'\s*,?\s*")

_HASPROP_RE = re.compile(r"hasProperty\s*\(\s*[^,]+,\s*(P[1-4]')\s*,\s*(V[1-4]')\s*\)")

_HASREL_RE = re.compile(r"hasRelationship\s*\(\s*[^,]+\s*,\s*[^,]+\s*,\s*(D[^,)]*)\s*\)")

# ── Template loading ────────────────────────────────────────────────────────

def load_template(path: Path) -> Tuple[str, str]:
    """
    Read an ASP template file and return (description, rule_text).

    Lines starting with '%' are comments that form the description.
    All other non-empty lines are the ASP rule text.
    """
    comments, rules = [], []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            comments.append(stripped.lstrip("% ").strip())
        else:
            rules.append(stripped)
    description = " ".join(comments)
    rule_text = " ".join(rules)
    return description, rule_text



# ── Placeholder discovery ───────────────────────────────────────────────────

def placeholders(rule: str) :
    """Return unique primed placeholders found in rule, in order of first appearance."""
    match_N = _N_RE.findall(rule)
    match_prop_val = _HASPROP_RE.finditer(rule) 
    match_rel = _HASREL_RE.findall(rule)
    result: list[str] = []
    pair_pv = []
    if match_N:
        for n in match_N:
            if n not in result:
                result.append(n)

    if match_prop_val:
        for match in match_prop_val:
            prop = match.group(1)
            value = match.group(2)
            if (prop, value) not in pair_pv:
                pair_pv.append((prop, value))
            if prop not in result:
                result.append(prop)
            if value not in result:
                result.append(value)
    
    if match_rel:
        for d in match_rel:
            if d not in result:
                result.append(d)
    
    return result, pair_pv


def inequalities(rule: str) -> Set[frozenset]:
    """Return set of {a, b} frozensets where a != b is required by the rule."""
    return {frozenset([a, b]) for a, b in _INEQ_RE.findall(rule)}


# ── Domain helpers ──────────────────────────────────────────────────────────

def _paired_p(v_ph: str, pair_pv) -> str:
    """Return the P placeholder paired with a V placeholder: V2' → P2'."""
    for (p,v) in pair_pv:
        if v == v_ph:
            return P
    


def _domain(PROPERTIES:Dict[str, List[str]],  ph: str, partial: Dict[str, str], pair_pv: List[Tuple[str, str]]) -> List[str]:
    """
    Return the value domain for a placeholder, given the partial assignment so far.
    For V placeholders, the domain is the property values of the paired P placeholder.
    """
    # V placeholder — depends on paired P
    for (prop, val) in pair_pv:
        if ph==val:
            p_ph = prop
    prop = partial.get(p_ph)
    if prop and prop in PROPERTIES:
        return list(PROPERTIES[prop])
    

def _violates(candidate: Dict[str, str], ineqs: Set[frozenset]) -> bool:
    """Return True if the candidate assignment violates any inequality constraint."""
    return any(
        candidate.get(a) == candidate.get(b)
        for pair in ineqs
        for a, b in [tuple(pair)]
        if a in candidate and b in candidate
    )


# ── Assignment generation ───────────────────────────────────────────────────

def sample_assignment(domain:str, rule: str, rng: random.Random) -> Dict[str, str]:
    """
    Sample one valid random assignment for all primed placeholders in rule.

    Placeholders are filled in order P → V → D → N, with V placeholders
    depending on their paired P. Explicit != constraints in the rule are enforced.

    Raises RuntimeError if a valid assignment cannot be found after many attempts.
    """
    domain_module = load_domain(domain)
    # Domain-specific vocabulary
    _ALL_PROPERTIES = domain_module.PROPERTIES
    RELATIONS = domain_module.DIRECTIONS
    
    PROPERTIES: Dict[str, List[str]] = {k: v for k, v in _ALL_PROPERTIES.items() if k != "size"}


    phs, pair_pv = placeholders(rule)
    ineqs = inequalities(rule)
    
    p_phs = [ph for ph in phs if ph.startswith("P")]
    v_phs = [ph for ph in phs if ph.startswith("V")]
    d_phs = [ph for ph in phs if ph.startswith("D")]
    n_phs = [ph for ph in phs if ph.startswith("N")]

    for _ in range(2000):
        asgn: Dict[str, str] = {}

        # Fill P placeholders
        valid = True
        for ph in p_phs:
            excluded = {asgn[o] for o in asgn if frozenset([ph, o]) in ineqs}
            
            choices = [p for p in PROPERTIES if p not in excluded]
            if not choices:
                valid = False
                break
            asgn[ph] = rng.choice(choices)
            
        if not valid:
            continue

        # Fill V placeholders (depend on paired P)
        for ph in v_phs:
            domain = _domain(PROPERTIES, ph, asgn, pair_pv)
            excluded = {asgn[o] for o in asgn if frozenset([ph, o]) in ineqs}
            choices = [v for v in domain if v not in excluded]
            if not choices:
                valid = False
                break
            asgn[ph] = rng.choice(choices)
        if not valid:
            continue

        # Fill D placeholders
        for ph in d_phs:
            #excluded = {asgn[o] for o in asgn if frozenset([ph, o]) in ineqs}
            choices = [d for d in RELATIONS]
            if not choices:
                valid = False
                break
            asgn[ph] = rng.choice(choices)
        if not valid:
            continue

        # Fill N placeholders
        for ph in n_phs:
            asgn[ph] = rng.choice(COUNTS)

        if all(ph in asgn for ph in phs) and not _violates(asgn, ineqs):
            return asgn

    raise RuntimeError(
        f"Could not find a valid assignment for rule after 2000 attempts.\n"
        f"Placeholders: {phs}\nInequalities: {ineqs}"
    )

# ── Rule instantiation ──────────────────────────────────────────────────────

def apply_assignment(domain:str, rule: str, assignment: Dict[str, str]) -> str:
    """
    Substitute all primed placeholders in rule with their assigned values.

    Longer placeholder names are replaced first to avoid partial matches
    (e.g. P1' before P1 if both existed, though all placeholders end with ').
    """
    domain_module = load_domain(domain)
    # Domain-specific vocabulary
    _ALL_PROPERTIES = domain_module.PROPERTIES
    
    PROPERTIES: Dict[str, List[str]] = {k: v for k, v in _ALL_PROPERTIES.items() if k != "size"}

    out = rule
    out = _REMOVE_INEQ_RE.sub("",out)
    # Remove a comma immediately before a period
    out = re.sub(r",\s*\.", ".", out)
    # Remove a comma immediately before a }
    out = re.sub(r",\s*}", "}", out)
    # Optional: clean up duplicate commas
    out = re.sub(r"\s*,\s*,\s*", ", ", out) 
    
    for ph in assignment:
        out = out.replace(ph, assignment[ph])
    
    return out


