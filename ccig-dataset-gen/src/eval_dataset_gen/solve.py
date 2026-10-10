"""
Clingo wrapper: solve ASP programs and sample grounded models.

The solve() function calls clingo on a complete ASP program and returns:
  - status: 'SAT', 'UNSAT', or 'TIMEOUT'
  - models: list of answer sets (each as a list of ground atoms)

The format_scene() helper structures ground atoms into a human-readable scene dict.
2"""

import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set


# ── Core solver ─────────────────────────────────────────────────────────────

def solve(
    program: str,
    *,
    n_models: int = 5,
    time_limit: int = 10,
    clingo_bin: str = "clingo",
) -> Tuple[str, List[List[str]]]:
    """
    Run clingo on the given ASP program.

    Args:
        program:     Complete ASP program string.
        n_models:    Maximum number of models to enumerate (0 = all).
        time_limit:  Wall-clock time limit in seconds.
        clingo_bin:  Name or path of the clingo executable.

    Returns:
        (status, models) where status is 'SAT', 'UNSAT', or 'TIMEOUT',
        and models is a list of answer sets (each answer set is a list of atom strings).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lp", delete=False, prefix="ccig_"
    ) as f:
        f.write(program)
        tmp_path = f.name

    cmd = [
        clingo_bin,
        str(n_models),
        tmp_path,
        f"--time-limit={time_limit}",
        "--outf=2",  # JSON output for clean parsing
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 5,
        )
        output_text = result.stdout
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return "TIMEOUT", []
    except FileNotFoundError:
        raise RuntimeError(
            f"clingo not found at '{clingo_bin}'. "
            "Install clingo (e.g. 'pip install clingo' or 'conda install -c potassco clingo')."
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return _parse_clingo_json(output_text)


def _parse_clingo_json(output: str) -> Tuple[str, List[List[str]]]:
    """Parse clingo's JSON output (--outf=2) into (status, models)."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        # Fall back to text parsing if JSON is malformed
        return _parse_clingo_text(output)

    result_str = data.get("Result", "")
    if "UNSATISFIABLE" in result_str:
        return "UNSAT", []
    if "TIME LIMIT" in result_str:
        return "TIMEOUT", []

    models: list[list[str]] = []
    for call in data.get("Call", []):
        for witness in call.get("Witnesses", []):
            models.append(witness.get("Value", []))

    status = "SAT" if models or "SATISFIABLE" in result_str else "UNKNOWN"
    return status, models


def _parse_clingo_text(output: str) -> Tuple[str, List[List[str]]]:
    """Fallback text parser for clingo stdout (used when JSON parsing fails)."""
    models: list[list[str]] = []
    current: list[str] | None = None

    for line in output.splitlines():
        if line.startswith("Answer:"):
            current = []
        elif current is not None and line.strip() and not any(
            line.startswith(p)
            for p in ("SATISFIABLE", "UNSATISFIABLE", "Models", "Time", "CPU", "Calls")
        ):
            current.extend(line.strip().split())
            models.append(current)
            current = None

    if "UNSATISFIABLE" in output:
        return "UNSAT", []
    if "TIME LIMIT" in output:
        return "TIMEOUT", []
    if "SATISFIABLE" in output or models:
        return "SAT", models
    return "UNKNOWN", []


# ── Scene formatter ─────────────────────────────────────────────────────────

_PROP_ATOM = re.compile(r"hasProperty\((\d+),(\w+),([\w_]+)\)")
_REL_ATOM  = re.compile(r"hasRelationship\((\d+),(\d+),([\w_]+)\)")


def format_scene(atoms: List[str]) -> Dict:
    """
    Parse a list of ground ASP atoms into a structured scene dict.

    Returns:
        {
          "objects": {
              "0": {"color": "red", "shape": "cube", "size": "small", "region": "region_1"},
              ...
          },
          "relations": [
              {"from": 0, "to": 1, "direction": "left"},
              ...
          ]
        }
    """
    objects: Dict[str, Dict[str, str]] = {}
    relations: List[Dict] = []

    for atom in atoms:
        m = _PROP_ATOM.match(atom)
        if m:
            obj_id, prop, val = m.group(1), m.group(2), m.group(3)
            objects.setdefault(obj_id, {})[prop] = val
            continue

        m = _REL_ATOM.match(atom)
        if m:
            relations.append({
                "from": int(m.group(1)),
                "to": int(m.group(2)),
                "direction": m.group(3),
            })

    return {"objects": objects, "relations": relations}
