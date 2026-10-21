"""Evaluate GenoScape's growth-medium extraction -- extract_media_composition (the
retrieval method save_carbon_source_media_records actually uses: rank paragraphs by
concentration-unit density, verify + extract from the best one) vs
extract_media_composition_whole_text (upload the entire paper in one call) -- on 5 real
papers that actually describe a growth medium's composition.

No hand-written ground truth, and no pre-built list of strains: this samples random
BacDive strains (no filtering on title or content -- whatever reference BacDive cites
is what the real pipeline would hit too) until it has found 5 DISTINCT references that
(a) resolve to real text via PMC or an actual PDF -- a scraped HTML landing page is
rejected; re-fetching the same DOI minutes apart was observed to flip between real
article text and mostly site-navigation chrome -- and (b) the retrieval method itself
confirms actually describes a growth medium, so every paper this evaluates genuinely
has media info to compare the two methods against, not just "paper resolved, but
describes nothing."

For each of those 5 papers: both methods' output is judged by an LLM that reads the
complete paper text itself, the exact instructions the extraction model was given, and
the output, and scores how faithfully that output reflects what the paper actually
says (see examples/judge.py:judge_media_extraction) -- no hand-written ground truth.

    python -m examples.eval_media

Writes data/eval_media_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from examples.build_ground_truth import _strain_name
from examples.Genoscape import load_env, make_model
from examples.judge import judge_media_extraction
from examples.my_tools import (EXTRACTION_SYSTEM, ToolError, _fetch_bacdive_record,
                               _resolve_reference_text, extract_media_composition,
                               extract_media_composition_whole_text, get_bacdive_carbon_source_growth)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "eval_media_results.json"
N_PAPERS = 5
SEED = 7  # fixed so the "random" sampling is reproducible
ID_RANGE = (1, 179500)  # BacDive-IDs are sparse above this; see examples/my_tools.py
MIN_TEXT_LEN = 2000  # long enough to be worth handing a judge, not an abstract stub
# Accept PMC full text or an actual PDF; reject a plain HTML landing-page fetch (see
# module docstring).
ACCEPTABLE_TEXT_SOURCES = {"pubmed_central", "pdf"}

JUDGE_MODEL_IDS = {"anthropic": "claude-opus-5", "openrouter": "anthropic/claude-opus-5"}
METHODS = {"whole_text": extract_media_composition_whole_text,
          "retrieval": extract_media_composition}


def run_method(name: str, fn, text: str) -> dict:
    usage_log: list[dict] = []
    try:
        result = fn(text, usage_log=usage_log)
    except ToolError as e:
        result = {"found": False, "media": [], "error": str(e)}
    tokens_in = sum(u["input_tokens"] for u in usage_log)
    tokens_out = sum(u["output_tokens"] for u in usage_log)
    return {"method": name, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "found": result.get("found", False), "media": result.get("media", []),
            "attempts": result.get("attempts"), "error": result.get("error")}


def _candidate_references(seed: int, id_range: tuple[int, int]):
    """Yield one dict per candidate for random BacDive-IDs (reproducibly, given `seed`)
    whose first carbon-source reference resolves to real text from PMC or a PDF -- no
    filtering on what that reference actually is. Runs forever; the caller decides when
    it has enough.

    Skips papers already yielded, keyed on doi/url rather than ref_id: BacDive gives
    each strain record its own reference @id even when several point at the identical
    paper (one widely-cited bulk-annotation paper turns up again and again this way),
    and the caller pays real extraction-model calls per candidate, so the same paper
    must not come back under a different @id."""
    rng = random.Random(seed)
    lo, hi = id_range
    tried, seen_papers = set(), set()
    while True:
        bid = rng.randint(lo, hi)
        if bid in tried:
            continue
        tried.add(bid)
        try:
            tests = get_bacdive_carbon_source_growth(bid)["tests"]
        except ToolError:
            continue  # invalid id, or no carbon-source data -- try another
        if not tests:
            continue
        record = _fetch_bacdive_record(bid)
        ref_id = tests[0]["ref"]
        try:
            text, provenance = _resolve_reference_text(bid, record, ref_id)
        except ToolError:
            continue
        if provenance["text_source"] not in ACCEPTABLE_TEXT_SOURCES or len(text) < MIN_TEXT_LEN:
            continue
        paper_key = (provenance.get("doi_or_url") or "").strip().lower() or f"ref:{ref_id}"
        if paper_key in seen_papers:
            continue
        seen_papers.add(paper_key)
        yield {"bacdive_id": bid, "strain_name": _strain_name(record), "ref_id": ref_id,
               "reference_title": provenance.get("reference_title"),
               "doi_or_url": provenance.get("doi_or_url"), "text": text}


def find_and_evaluate_papers(judge_model, n: int = N_PAPERS, seed: int = SEED,
                            id_range: tuple[int, int] = ID_RANGE,
                            out_path: Path = RESULTS_PATH) -> list[dict]:
    """Sample candidate papers until `n` are found where the retrieval method actually
    reports finding a medium -- confirming the paper genuinely has media info, not just
    that it resolved -- then run both methods and judge each against the paper. Writes
    `out_path` after every accepted paper."""
    results = []
    for cand in _candidate_references(seed, id_range):
        text = cand["text"]
        print(f"\n=== candidate: {cand['strain_name']} (bacdive_id {cand['bacdive_id']}), "
             f"{cand['doi_or_url']} ({cand['reference_title']!r}) ===")
        row = {k: v for k, v in cand.items() if k != "text"}
        row["text_len"] = len(text)
        for method_name, fn in METHODS.items():
            print(f"  running {method_name}...")
            row[method_name] = run_method(method_name, fn, text)
        if not row["retrieval"]["found"]:
            print("  retrieval method found no medium here -- doesn't have usable media "
                 f"info, skipping (not counted toward the {n})")
            continue
        for method_name in METHODS:
            print(f"  judging {method_name}...")
            score = judge_media_extraction(judge_model, cand["strain_name"], EXTRACTION_SYSTEM,
                                           text, {"found": row[method_name]["found"],
                                                  "media": row[method_name]["media"]})
            row[method_name]["score"] = score
        results.append(row)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  accepted ({len(results)}/{n})")
        if len(results) >= n:
            break
    return results


def print_summary(results: list[dict]) -> None:
    header = f"{'strain':<35}{'method':<12}{'accuracy':>10}{'tokens':>9}{'found':>7}"
    print(header)
    print("-" * len(header))
    for row in results:
        for method_name in METHODS:
            m = row[method_name]
            acc = m["score"].get("accuracy")
            acc_str = "-" if acc is None else f"{acc:.2f}"
            tokens = m["input_tokens"] + m["output_tokens"]
            print(f"{row['strain_name'][:34]:<35}{method_name:<12}{acc_str:>10}{tokens:>9}{str(m['found']):>7}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=N_PAPERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--provider", choices=["openrouter", "anthropic"],
                        default="openrouter" if os.environ.get("OPENROUTER_API_KEY") else "anthropic")
    args = parser.parse_args()

    judge_model = make_model(args.provider, JUDGE_MODEL_IDS[args.provider])
    results = find_and_evaluate_papers(judge_model, n=args.n, seed=args.seed, out_path=Path(args.out))

    print(f"\nwrote {args.out}\n")
    print_summary(results)


if __name__ == "__main__":
    main()
