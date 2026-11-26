"""Crisp prompt templates used by textual inversion.

The templates are read from the existing new_crisp data module so there is a
single source of truth. Current PixArt Stage-0 route uses the single-placeholder
list; dual templates are reserved for checkpoints migrated from Stage-0.
"""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}

def load_templates(name: str = "imagenet_templates_small"):
    if name in _CACHE:
        return _CACHE[name]
    tree = ast.parse((ROOT / "ldm" / "data" / "personalized.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    out = ast.literal_eval(node.value)
                    _CACHE[name] = out
                    return out
    raise RuntimeError(f"{name} not found")

def canonical_prompt(placeholder: str, template_index: int = 0) -> str:
    return load_templates("imagenet_templates_small")[template_index].format(placeholder)

def random_prompt(placeholder: str, rng) -> str:
    return rng.choice(load_templates("imagenet_templates_small")).format(placeholder)

def dual_prompt(placeholder: str, dual_placeholder: str, rng) -> str:
    return rng.choice(load_templates("imagenet_dual_templates_small")).format(placeholder, dual_placeholder)
