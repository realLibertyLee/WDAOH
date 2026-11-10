"""Fetch and load the exact upstream architecture definitions used by the study."""

import ast
import hashlib
import math
import urllib.request
from functools import lru_cache
from pathlib import Path

from wdaoh.protocol import digest, settings


def acquire_sources(work: Path, model: str):
    """Keep upstream source outside the checkout and verify each file against the pinned commit and digest."""
    spec = settings("upstream")[model]
    project = spec["repository"].removeprefix("https://github.com/")
    base = work / "upstream" / spec["revision"]
    for entry in spec["sources"]:
        path = base / entry["path"]
        if path.is_file() and digest(path) == entry["sha256"]:
            continue
        url = f"https://raw.githubusercontent.com/{project}/{spec['revision']}/{entry['path']}"
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read(1 << 20)
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed upstream source: {url}")
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(body)
    return base


@lru_cache
def architecture(work, model):
    """Load verified class bodies in a private namespace, keeping upstream projects' identical module names independent."""
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional

    spec = settings("upstream")[model]
    names = {
        "torch": torch,
        "nn": nn,
        "np": np,
        "math": math,
        "sqrt": math.sqrt,
        "F": functional,
        "__name__": "wdaoh_upstream_" + model,
    }
    for entry in spec["sources"]:
        path = work / "upstream" / spec["revision"] / entry["path"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(
                f"Missing or changed official file {entry['path']}. Run model-sources --model {model}."
            )
        tree = ast.parse(path.read_bytes(), filename=str(path))
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        module = ast.Module(
            body=[definitions[key] for key in entry["definitions"]], type_ignores=[]
        )
        exec(compile(module, str(path), "exec"), names)
    return names["Model"]
