"""Regression test for the shipped NameError: `clipplan` is not defined.

pipeline.py called ``clipplan.build_sync_beats(...)`` in Step D but the
``from . import ...`` list never imported ``clipplan`` -- so every run
died at render time, after all the expensive steps (narration included)
had already succeeded. This test statically asserts that every recap
submodule a module *references* as ``name.`` is actually imported by it,
for every module in the package. No third-party deps required.

Run:  python tests/test_pipeline_imports.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECAP = ROOT / "recap"


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and not node.module:      # from . import a, b
                for a in node.names:
                    names.add(a.asname or a.name)
            elif node.level == 1 and node.module:        # from .x import y
                names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def test_submodule_references_are_imported():
    submods = {p.stem for p in RECAP.glob("*.py")}
    failures = []
    for path in sorted(RECAP.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = _imported_names(tree)
        # names bound locally (functions/classes/assignments) are fine too
        bound = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        # locals/params named like a module (e.g. a `beats` list) are not
        # submodule references
        bound |= {n.id for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        used = {n.value.id for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        missing = sorted(u for u in used
                         if u in submods and u not in imported and u not in bound
                         and u != path.stem)
        if missing:
            failures.append(f"{path.name}: uses {missing} but never imports them")
    assert not failures, "\n".join(failures)
    print("ok: every recap submodule referenced anywhere is imported")


if __name__ == "__main__":
    test_submodule_references_are_imported()
    print("\nall pipeline import tests passed")
