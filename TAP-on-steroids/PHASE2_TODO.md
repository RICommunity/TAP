# Phase 2 — Memory Layer: Implementation Tracker

Single source of truth for what was built, why, and what's left. Companion to
`ROADMAP_AGENT.md` (§Phase 2, lines 32–43).

**Core idea:** make stateless TAP *stateful* by attaching a vector memory of
past successful attacks. Before a run we retrieve the top-k most relevant past
jailbreaks and inject them as few-shot examples into the attacker's system
prompt; after a run we write the new successes back. Learning = ASR rising
across runs against a fixed target, no model weights changed.

---

## Status: Phase 2 build COMPLETE ✅ (experiments = Phase 3, not yet run)

| # | Task (from roadmap) | Status | Where |
|---|---------------------|--------|-------|
| 1 | Build the store `(prompt, embedding, score, goal_category, strategy_tag)` | ✅ | `stupid-TAP/memory.py` |
| 2 | Write path: store successful prompts (score ≥ threshold) after each run | ✅ | `main_TAP.py` write-path + `AttackMemory.add_successes` |
| 3 | Read path: embed goal, retrieve top-k, inject as few-shot | ✅ | `main_TAP.py` read-path + `AttackMemory.retrieve` + `system_prompts.get_attacker_system_prompt_with_memory` |
| 4a | Mechanism 1 — plain retrieval (similarity) | ✅ | `retrieve(mode="plain")` |
| 4b | Mechanism 2 — score-weighted retrieval (similarity × score) | ✅ | `retrieve(mode="score_weighted")` |
| 4c | Mechanism 3 — strategy tagging (optional) | ⬜ deferred | hook present (`strategy_tag` metadata) |

---

## What was implemented

### `stupid-TAP/language_models.py`
- `OpenRouterEmbedder` — client for OpenRouter's `/embeddings` endpoint
  (`openai/text-embedding-3-small`, 1536-dim, batched). Vectors are produced
  here and stored directly, so the vector store needs no local embedding model.

### `stupid-TAP/memory.py`  (new)
- `VectorStore` — minimal persistent cosine-similarity store backed by NumPy
  (`store.npz` for vectors + `store.jsonl` for records). Brute-force search;
  instant at our scale (hundreds of vectors).
- `AttackMemory` — the public API:
  - `add_successes(items, goal, target_model, goal_category, write_threshold, pass_index)` → write path (dedup by `(target, goal, prompt)`, only score ≥ threshold).
  - `retrieve(goal, k, mode, target_model, exclude_goal)` → read path (plain / score_weighted; optional same-goal exclusion; optional per-target filtering).
  - `count()`, `reset()`.

### `stupid-TAP/system_prompts.py`
- `format_memory_examples(retrieved)` and `get_attacker_system_prompt_with_memory(...)`.
- **Guarantee:** with no retrieved examples the output is byte-identical to the
  baseline `get_attacker_system_prompt`, so *memory-OFF reproduces the baseline exactly*.

### `stupid-TAP/main_TAP.py`
- Read path before the tree loop; write path after it; a `memory_meta_iter_<i>.json`
  sidecar per run (memory size, #retrieved, #written, retrieved sources) for Phase-3 analysis.
- New flags (all no-ops unless `--use-memory`): `--use-memory`, `--memory-dir`,
  `--memory-collection`, `--retrieval-mode {plain,score_weighted}`, `--retrieval-k`,
  `--memory-exclude-same-goal`, `--memory-any-target`, `--memory-write/--no-memory-write`,
  `--memory-write-threshold`, `--embedding-model`, `--goal-category`, `--pass-index`.

### `TAP-on-steroids/memory_experiment.py`  (orchestrator)
- Runs conditions × N sequential passes over the behaviour set (shells out to
  `main_TAP.py`, same as the baseline notebook). Memory persists per condition and
  is reset to empty before pass 0 (cold start).
- Registered conditions: `baseline`, `plain_same`, `plain_xgoal`,
  `score_weighted_same`, `score_weighted_xgoal`.

### `TAP-on-steroids/analysis.py`
- `analyze(out_dir)` → per-prompt metrics, ASR-over-passes curve,
  queries-to-success curve, ablation table, and (if matplotlib present) `learning_curves.png`.
- Metrics mirror the baseline `collect_metrics` exactly (jailbroken = any judge==10;
  n_queries = rows logged; max_judge_score).

---

## Design decisions (confirmed with the user)

- **Embeddings:** OpenRouter `/embeddings`, `openai/text-embedding-3-small`. (OpenRouter
  now exposes embeddings; only `OPENROUTER_API_KEY` is set in `.env`, no OpenAI key needed.)
- **Success/write threshold:** only full jailbreaks (`judge score == 10`).
- **Run semantics:** empty memory → N sequential passes; memory grows each pass → learning curve.
- **Retrieval scope:** both include- and exclude-same-goal supported as an ablation toggle.

## ⚠️ Open design point to decide before Phase-3 runs

**Score-weighted vs. write threshold.** Because we only store score-10 prompts,
every stored item has the *same* score, so `similarity × score` collapses onto
plain similarity — the `score_weighted` ablation would be a no-op (verified: its
ranking equals plain's on the current baseline data).

To make mechanism 2 a *real* ablation, store a **range** of scores. The code
already supports this: set a lower `--memory-write-threshold` (e.g. `6`) for the
score-weighted condition, and the actual judge score is stored and used in ranking.
The ASR "success" definition (== 10) is intentionally kept separate from the
write threshold, so lowering the write threshold does **not** change the headline metric.

Decision needed: **what write threshold for the score-weighted arm?** (recommend `>= 6`).

---

## Environment notes (important)

- **Backend: ChromaDB (default).** `memory.py` stores vectors in a persistent
  Chroma collection (`hnsw:space=cosine`); embeddings come from the OpenRouter
  embeddings API (Chroma is used purely as the index). A dependency-free
  **NumPy** store implements the same interface and is selected with
  `--memory-backend numpy` (or used automatically if `chromadb` fails to import).
- **chromadb + protobuf/pydantic.** `chromadb==1.5.9` bumps `protobuf` to 7 and
  `pydantic` to 2. This is safe here: we no longer import `wandb`, and only use
  `fastchat.conversation`, which works under pydantic 2. `pip` prints harmless
  "incompatible" warnings for the old `wandb`/`fschat` pins.
  NB: the earlier "chromadb segfaults" finding was a **sandbox artifact** (the
  same sandbox also segfaults plain `import numpy`); Chroma runs fine normally.
- **wandb removed from the logger.** `loggers.py` no longer imports wandb (kept the
  class name `WandBLogger` for compatibility); it still writes the same
  `iter_<i>_df` parquet/csv files that `collect_metrics` consumes.
- **`pyarrow` pinned to `16.1.0`.** Newer pyarrow ships numpy-2-ABI wheels that are
  incompatible with the pinned `numpy==1.26.3`.
- The `.venv` was rebuilt cleanly from `requirements.txt` on the pyenv 3.11.15 base.

---

## How to run (Phase 3)

```bash
# From TAP-on-steroids/ (sibling of stupid-TAP)
python memory_experiment.py \
  --target claude-3-haiku --attacker gpt-4o-mini --evaluator gpt-4o-mini \
  --n-prompts 15 --n-passes 3 \
  --conditions baseline plain_same score_weighted_same

python analysis.py memory_results/memory_results_<date>
```

Repeat per target (`gpt-4.1`, `gemini-2.5-flash`) to match the three baseline combos.

---

## Next steps
- [ ] Decide the score-weighted write threshold (see open design point).
- [ ] (Optional) add strategy tagging (mechanism 3).
- [ ] Run Phase-3 experiments across the 3 targets and generate curves + ablation table.
- [ ] Consider warm-seeding memory from existing `baseline_results` as an extra arm.
