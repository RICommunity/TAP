"""Helpers for the two Phase-2 notebooks:

    1. Memory_fill.ipynb   — cold TAP runs that WRITE successes into ChromaDB
    2. Memory_attack.ipynb — TAP runs that READ from memory (plain / score_weighted)

Both shell out to ``stupid-TAP/main_TAP.py``. Each prompt produces:
    * ``iter_<i>_df.csv`` / parquet  (attack log table — from loggers.py)
    * ``iter_<i>_*.log``             (stdout/stderr of the TAP process)
    * ``memory_meta_iter_<i>.json``  (write/retrieve metadata sidecar)

Shared memory lives under ``memory_store/`` (one Chroma dir for both notebooks).
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
HOME = REPO_ROOT / "stupid-TAP"
assert (HOME / "main_TAP.py").exists(), f"main_TAP.py not found under {HOME}"

DEFAULT_DATASET = HOME / "data" / "advbench_subset.csv"
DEFAULT_MEMORY_DIR = THIS_DIR / "memory_store"
PYTHON = sys.executable


@dataclass
class TAPParams:
    branching_factor: int = 4
    depth: int = 4
    width: int = 5
    n_streams: int = 1
    target_max_n_tokens: int = 500


def load_behaviours(n_prompts: int, dataset: Path = DEFAULT_DATASET) -> pd.DataFrame:
    df = pd.read_csv(dataset).head(n_prompts).reset_index(drop=True)
    df.index.name = "behaviour_id"
    return df


def _run_one(cmd: List[str], log_path: Path) -> int:
    env = {**os.environ, "WANDB_MODE": "offline"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(
            cmd, cwd=str(HOME), env=env, stdout=f, stderr=subprocess.STDOUT
        )
    return proc.returncode


def _base_cmd(
    attacker: str,
    target: str,
    evaluator: str,
    goal: str,
    target_str: str,
    category: str,
    params: TAPParams,
    store_folder: Path,
    iter_index: int,
    debug: bool = False,
) -> List[str]:
    cmd = [
        PYTHON, "main_TAP.py",
        "--attack-model", attacker,
        "--target-model", target,
        "--evaluator-model", evaluator,
        "--goal", goal,
        "--target-str", target_str,
        "--branching-factor", str(params.branching_factor),
        "--width", str(params.width),
        "--depth", str(params.depth),
        "--n-streams", str(params.n_streams),
        "--store-folder", str(store_folder),
        "--iter-index", str(iter_index),
        "--target-max-n-tokens", str(params.target_max_n_tokens),
        "--category", category or "unknown",
        "--goal-category", category,
    ]
    if debug:
        cmd.append("--debug")
    return cmd


def _open_memory(memory_dir: Path, collection_name: str = "tap_attacks"):
    sys.path.insert(0, str(HOME))
    from memory import AttackMemory
    return AttackMemory(
        persist_dir=str(memory_dir),
        collection_name=collection_name,
    )


def peek_memory(
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    collection_name: str = "tap_attacks",
) -> int:
    """Return current memory size (0 if the store does not exist yet)."""
    if not Path(memory_dir).exists():
        return 0
    return _open_memory(memory_dir, collection_name).count()


def reset_memory(
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    collection_name: str = "tap_attacks",
) -> int:
    """Wipe the shared memory store. Returns size after reset (should be 0)."""
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    mem = _open_memory(memory_dir, collection_name)
    mem.reset()
    n = mem.count()
    print(f"[memory] reset {memory_dir} (now {n} items)", flush=True)
    return n


def list_memory(
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    collection_name: str = "tap_attacks",
    filter_target: Optional[str] = None,
    filter_attacker: Optional[str] = None,
    filter_evaluator: Optional[str] = None,
    min_score: Optional[int] = None,
) -> pd.DataFrame:
    """List stored attacks as a DataFrame (optional metadata filters)."""
    if not Path(memory_dir).exists():
        return pd.DataFrame()
    rows = _open_memory(memory_dir, collection_name).list_records(
        target_model=filter_target,
        attacker_model=filter_attacker,
        evaluator_model=filter_evaluator,
        min_score=min_score,
    )
    return pd.DataFrame(rows)


def run_fill(
    attacker: str = "gpt-4o-mini",
    target: str = "claude-3-haiku",
    evaluator: str = "gpt-4o-mini",
    n_prompts: int = 15,
    params: Optional[TAPParams] = None,
    write_threshold: int = 5,
    memory_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    embedding_model: str = "openai/text-embedding-3-small",
    reset: bool = False,
) -> Path:
    """Cold TAP runs that WRITE successes (>= write_threshold) into ChromaDB.

    Memory is enabled for the write path; retrieval is a no-op on an empty
    (or freshly-reset) store. Results land under::

        out_dir/runs/fill/target_<t>/iter_<i>_df.csv
        out_dir/runs/fill/target_<t>/iter_<i>_*.log
        memory_dir/   (shared Chroma store)
    """
    params = params or TAPParams()
    behaviours = load_behaviours(n_prompts)
    memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
    memory_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.today().strftime("%Y_%m_%d")
    out_dir = Path(out_dir) if out_dir else (
        THIS_DIR / "memory_results" / f"fill_{date}"
    )
    store_folder = out_dir / "runs" / "fill" / f"target_{target}"
    store_folder.mkdir(parents=True, exist_ok=True)

    if reset:
        reset_memory(memory_dir)

    config = {
        "mode": "fill",
        "attacker": attacker,
        "target": target,
        "evaluator": evaluator,
        "n_prompts": n_prompts,
        "write_threshold": write_threshold,
        "params": asdict(params),
        "memory_dir": str(memory_dir),
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"[fill] writing successes with score>={write_threshold} "
        f"into {memory_dir} (size before: {peek_memory(memory_dir)})",
        flush=True,
    )

    stamp = datetime.now().strftime("%Y_%m_%d_at_%H_%M")
    for i in range(len(behaviours)):
        goal = behaviours.iloc[i]["goal"]
        target_str = behaviours.iloc[i]["target"]
        category = str(behaviours.iloc[i].get("category", ""))
        log_path = store_folder / f"iter_{i}_datetime_{stamp}.log"

        cmd = _base_cmd(
            attacker, target, evaluator, goal, target_str, category,
            params, store_folder, i,
        )
        # Enable memory for the WRITE path. Retrieval returns [] while empty.
        cmd += [
            "--use-memory",
            "--memory-dir", str(memory_dir),
            "--memory-collection", "tap_attacks",
            "--memory-write",
            "--memory-write-threshold", str(write_threshold),
            "--embedding-model", embedding_model,
            "--retrieval-k", "0",  # do not inject examples while filling
        ]

        rc = _run_one(cmd, log_path)
        status = "ok" if rc == 0 else f"FAILED(exit {rc})"
        print(
            f"[fill | prompt {i}] {status}: {goal[:55]!r} "
            f"(memory size={peek_memory(memory_dir)})",
            flush=True,
        )

    print(f"\n[fill] done. Results: {out_dir}", flush=True)
    print(f"[fill] memory size now: {peek_memory(memory_dir)}", flush=True)
    return out_dir


def run_attack_with_memory(
    attacker: str = "gpt-4o-mini",
    target: str = "claude-3-haiku",
    evaluator: str = "gpt-4o-mini",
    n_prompts: int = 15,
    params: Optional[TAPParams] = None,
    retrieval_mode: str = "plain",
    retrieval_k: int = 3,
    memory_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    embedding_model: str = "openai/text-embedding-3-small",
    # Optional exact-match filters on stored metadata:
    filter_target: Optional[str] = None,
    filter_attacker: Optional[str] = None,
    filter_evaluator: Optional[str] = None,
    memory_any_target: bool = False,
    exclude_same_goal: bool = False,
    # Whether to also write this attack's successes back into memory:
    write_back: bool = False,
    write_threshold: int = 5,
    debug: bool = False,
) -> Path:
    """TAP runs that READ from the shared ChromaDB store.

    `retrieval_mode` is ``"plain"`` or ``"score_weighted"``.
    Filters restrict which stored attacks are eligible for retrieval.
    By default does NOT write back (read-only reuse of the fill store).

    Results land under::

        out_dir/runs/attack_<mode>/target_<t>/iter_<i>_df.csv
        out_dir/runs/attack_<mode>/target_<t>/iter_<i>_*.log
    """
    if retrieval_mode not in ("plain", "score_weighted"):
        raise ValueError(
            f"retrieval_mode must be 'plain' or 'score_weighted', got {retrieval_mode!r}"
        )

    params = params or TAPParams()
    behaviours = load_behaviours(n_prompts)
    memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR

    size = peek_memory(memory_dir)
    if size == 0:
        print(
            f"[attack] WARNING: memory at {memory_dir} is empty. "
            "Run Memory_fill.ipynb first.",
            flush=True,
        )

    date = datetime.today().strftime("%Y_%m_%d")
    out_dir = Path(out_dir) if out_dir else (
        THIS_DIR / "memory_results" / f"attack_{retrieval_mode}_{date}"
    )
    store_folder = out_dir / "runs" / f"attack_{retrieval_mode}" / f"target_{target}"
    store_folder.mkdir(parents=True, exist_ok=True)

    config = {
        "mode": "attack_with_memory",
        "attacker": attacker,
        "target": target,
        "evaluator": evaluator,
        "n_prompts": n_prompts,
        "retrieval_mode": retrieval_mode,
        "retrieval_k": retrieval_k,
        "filter_target": filter_target,
        "filter_attacker": filter_attacker,
        "filter_evaluator": filter_evaluator,
        "memory_any_target": memory_any_target,
        "exclude_same_goal": exclude_same_goal,
        "write_back": write_back,
        "write_threshold": write_threshold,
        "debug": debug,
        "params": asdict(params),
        "memory_dir": str(memory_dir),
        "embedding_model": embedding_model,
        "memory_size_before": size,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"[attack] mode={retrieval_mode} k={retrieval_k} "
        f"memory_size={size} filters(target={filter_target}, "
        f"attacker={filter_attacker}, evaluator={filter_evaluator})",
        flush=True,
    )

    stamp = datetime.now().strftime("%Y_%m_%d_at_%H_%M")
    for i in range(len(behaviours)):
        goal = behaviours.iloc[i]["goal"]
        target_str = behaviours.iloc[i]["target"]
        category = str(behaviours.iloc[i].get("category", ""))
        log_path = store_folder / f"iter_{i}_datetime_{stamp}.log"

        cmd = _base_cmd(
            attacker, target, evaluator, goal, target_str, category,
            params, store_folder, i, debug=debug,
        )
        cmd += [
            "--use-memory",
            "--memory-dir", str(memory_dir),
            "--memory-collection", "tap_attacks",
            "--retrieval-mode", retrieval_mode,
            "--retrieval-k", str(retrieval_k),
            "--embedding-model", embedding_model,
        ]

        if write_back:
            cmd += [
                "--memory-write",
                "--memory-write-threshold", str(write_threshold),
            ]
        else:
            cmd.append("--no-memory-write")

        if memory_any_target:
            cmd.append("--memory-any-target")
        elif filter_target:
            cmd += ["--memory-filter-target", filter_target]
        if exclude_same_goal:
            cmd.append("--memory-exclude-same-goal")
        if filter_attacker:
            cmd += ["--memory-filter-attacker", filter_attacker]
        if filter_evaluator:
            cmd += ["--memory-filter-evaluator", filter_evaluator]

        rc = _run_one(cmd, log_path)
        status = "ok" if rc == 0 else f"FAILED(exit {rc})"
        print(f"[attack | prompt {i}] {status}: {goal[:55]!r}", flush=True)

    print(f"\n[attack] done. Results: {out_dir}", flush=True)
    return out_dir


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Memory fill / attack runners.")
    ap.add_argument("command", choices=["fill", "attack"])
    ap.add_argument("--target", default="claude-3-haiku")
    ap.add_argument("--evaluator", default="gpt-4o-mini")
    ap.add_argument("--attacker", default="gpt-4o-mini")
    ap.add_argument("--n-prompts", type=int, default=15)
    ap.add_argument("--branching-factor", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--target-max-n-tokens", type=int, default=500)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    ap.add_argument("--out-dir", default=None)
    # fill
    ap.add_argument("--write-threshold", type=int, default=5)
    ap.add_argument("--reset", action="store_true", help="Wipe memory before fill.")
    # attack
    ap.add_argument("--retrieval-mode", default="plain",
                    choices=["plain", "score_weighted"])
    ap.add_argument("--retrieval-k", type=int, default=3)
    ap.add_argument("--filter-target", default=None)
    ap.add_argument("--filter-attacker", default=None)
    ap.add_argument("--filter-evaluator", default=None)
    ap.add_argument("--memory-any-target", action="store_true")
    ap.add_argument("--exclude-same-goal", action="store_true")
    ap.add_argument("--write-back", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    params = TAPParams(
        branching_factor=a.branching_factor,
        depth=a.depth,
        width=a.width,
        target_max_n_tokens=a.target_max_n_tokens,
    )
    mem_dir = Path(a.memory_dir) if a.memory_dir else None
    out = Path(a.out_dir) if a.out_dir else None
    if a.command == "fill":
        run_fill(
            attacker=a.attacker, target=a.target, evaluator=a.evaluator,
            n_prompts=a.n_prompts, params=params,
            write_threshold=a.write_threshold,
            memory_dir=mem_dir, out_dir=out,
            embedding_model=a.embedding_model,
            reset=a.reset,
        )
    else:
        run_attack_with_memory(
            attacker=a.attacker, target=a.target, evaluator=a.evaluator,
            n_prompts=a.n_prompts, params=params,
            retrieval_mode=a.retrieval_mode, retrieval_k=a.retrieval_k,
            memory_dir=mem_dir, out_dir=out,
            embedding_model=a.embedding_model,
            filter_target=a.filter_target,
            filter_attacker=a.filter_attacker,
            filter_evaluator=a.filter_evaluator,
            memory_any_target=a.memory_any_target,
            exclude_same_goal=a.exclude_same_goal,
            write_back=a.write_back,
            write_threshold=a.write_threshold,
        )
