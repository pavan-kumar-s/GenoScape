"""Evaluate GenoScape's growth-medium extraction -- extract_media_composition (the
retrieval method save_carbon_source_media_records actually uses: rank paragraphs by
concentration-unit density, verify + extract from the best one, retry the next-best if
nothing found) vs extract_media_composition_whole_text (upload the entire paper in one
call) -- on real papers that actually describe how a microbe was grown.

Papers come straight from PubMed Central's open-access subset, searched for species-
description papers ("... sp. nov. ..."), which by convention state the medium the
organism was characterized on. No BacDive involvement: going through BacDive's cited
references turned out to be a dead end for this -- most resolve only to paywalled
landing pages, and the one reference that reliably resolves is a single over-cited
database-mining paper that describes no medium at all. PMC full text is genuine and
reproducible, and is fetched through the very same get_pubmed_reference_text the
pipeline uses.

A paper is only evaluated if a judge call first confirms it really does state a
medium's composition (judge_paper_has_media). That gate deliberately does NOT consult
either extraction method: gating on the retrieval method's own "found" biased the eval
both ways -- it admitted that method's false positives, and it excluded the most
valuable case of all, a paper that really does contain a recipe which the method
missed.

Scoring uses no hand-written ground truth: for each paper, an LLM judge reads the
complete paper text itself, the exact instructions the extraction model was given, and
each method's output, and scores how faithfully that output reflects what the paper
actually says (examples/judge.py:judge_media_extraction).

    python -m examples.eval_media
    python -m examples.eval_media --n 3

Writes data/eval_media_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from examples.Genoscape import load_env, make_model
from examples.judge import judge_media_extraction, judge_paper_has_media
from examples.my_tools import (EXTRACTION_SYSTEM, ToolError, _ncbi_get,
                               extract_media_composition,
                               extract_media_composition_whole_text,
                               get_pubmed_reference_text)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "eval_media_results.json"
N_PAPERS = 5
MIN_TEXT_LEN = 2000  # long enough to be worth handing a judge, not an abstract stub
PAGE_SIZE = 50  # PMC search results fetched per page
MAX_PAGES = 10  # give up rather than paging forever

PMC_SEARCH_TERM = '"sp. nov."[Title] AND bacterium[Title/Abstract] AND open access[filter]'
# The binomial immediately BEFORE "sp. nov."/"gen. nov." anywhere in the title, e.g.
# "Streptomyces nigra sp. nov. Is a Novel Actinobacterium ..." -> "Streptomyces nigra".
# Anchoring at the title start instead matched the first two words of any title the
# search returned ("Complete genome of ...", "Spatial microbiome ...") and produced
# organism names that are not organisms.
SPECIES_NAME_RE = re.compile(r"\b([A-Z][a-z]+\s+[a-z]{3,})\s+(?:sp|gen)\.\s*nov\.")

JUDGE_MODEL_IDS = {"anthropic": "claude-opus-5", "openrouter": "anthropic/claude-opus-5"}
METHODS = {"whole_text": extract_media_composition_whole_text,
          "retrieval": extract_media_composition}


def _candidate_papers(page_size: int = PAGE_SIZE, max_pages: int = MAX_PAGES):
    """Yield {organism, doi, pmcid, title, text} for open-access PMC species-description
    papers whose full text actually came back. Runs until PMC's results are exhausted;
    the caller decides when it has enough."""
    seen_dois = set()
    for page in range(max_pages):
        search = json.loads(_ncbi_get("esearch", db="pmc", retmode="json", term=PMC_SEARCH_TERM,
                                      retstart=page * page_size, retmax=page_size))
        ids = (search.get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return  # exhausted the search
        summary = json.loads(_ncbi_get("esummary", db="pmc", retmode="json", id=",".join(ids)))
        result = summary.get("result") or {}
        for uid in result.get("uids") or []:
            doc = result.get(uid) or {}
            doi = next((a["value"] for a in doc.get("articleids", [])
                       if a.get("idtype") == "doi"), None)
            title = re.sub(r"<[^>]+>", "", doc.get("title") or "").strip()
            if not doi or doi in seen_dois or not title:
                continue
            match = SPECIES_NAME_RE.search(title)
            if not match:
                continue  # no "<Genus species> sp. nov." in the title -- can't name the organism
            seen_dois.add(doi)
            try:
                ref = get_pubmed_reference_text(doi)
            except ToolError:
                continue
            text = ref.get("text") or ""
            if len(text) < MIN_TEXT_LEN:
                continue
            yield {"organism": match.group(1), "doi": doi, "pmcid": ref.get("pmcid"),
                   "title": title, "text": text}


def run_method(name: str, fn, text: str, organism: str) -> dict:
    usage_log: list[dict] = []
    try:
        result = fn(text, usage_log=usage_log, strain_name=organism)
    except ToolError as e:
        result = {"found": False, "media": [], "error": str(e)}
    tokens_in = sum(u["input_tokens"] for u in usage_log)
    tokens_out = sum(u["output_tokens"] for u in usage_log)
    return {"method": name, "input_tokens": tokens_in, "output_tokens": tokens_out,
            "found": result.get("found", False), "media": result.get("media", []),
            "attempts": result.get("attempts"), "note": result.get("note"),
            "error": result.get("error")}


def find_and_evaluate_papers(judge_model, n: int = N_PAPERS,
                            out_path: Path = RESULTS_PATH) -> list[dict]:
    """Walk candidate papers until `n` have passed the has-media gate, running both
    extraction methods on each and judging both outputs against the paper itself.
    Writes `out_path` after every accepted paper, so an interrupted run keeps its
    results. The gate runs BEFORE either extraction, so a paper with no recipe costs
    one cheap judge call instead of two full extractions."""
    results = []
    for paper in _candidate_papers():
        text, organism = paper["text"], paper["organism"]
        print(f"\n=== {organism} | {paper['doi']} | {paper['pmcid']} | {len(text)} chars ===", flush=True)
        print(f"    {paper['title'][:110]}", flush=True)
        print("  checking whether this paper states a medium composition at all...", flush=True)
        gate = judge_paper_has_media(judge_model, organism, text)
        if gate.get("has_media") is None:
            # Distinct from a clean "no": the gate call itself failed to produce parseable
            # JSON, so this paper is unjudged, not rejected -- don't let that read as a
            # verdict on the paper.
            print(f"  SKIPPING UNJUDGED -- gate reply was not parseable JSON: "
                 f"{(gate.get('raw_reply') or '')[:200]!r}", flush=True)
            continue
        if not gate["has_media"]:
            print(f"  skipping, no composition stated: {gate.get('reason')}", flush=True)
            continue
        print(f"  has media info: {gate.get('reason')}", flush=True)

        row = {k: v for k, v in paper.items() if k != "text"}
        row["text_len"] = len(text)
        row["has_media_gate"] = gate
        for method_name, fn in METHODS.items():
            print(f"  running {method_name}...", flush=True)
            row[method_name] = run_method(method_name, fn, text, organism)
            print(f"    found={row[method_name]['found']} "
                 f"media_entries={len(row[method_name]['media'])} "
                 f"tokens={row[method_name]['input_tokens'] + row[method_name]['output_tokens']}", flush=True)
        for method_name in METHODS:
            print(f"  judging {method_name}...", flush=True)
            score = judge_media_extraction(judge_model, organism, EXTRACTION_SYSTEM, text,
                                           {"found": row[method_name]["found"],
                                            "media": row[method_name]["media"]})
            row[method_name]["score"] = score
            print(f"    accuracy={score.get('accuracy')} reason={score.get('reason')!r}", flush=True)
        results.append(row)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  accepted ({len(results)}/{n})")
        if len(results) >= n:
            break
    return results


def print_summary(results: list[dict]) -> None:
    header = f"{'organism':<32}{'method':<12}{'accuracy':>10}{'tokens':>9}{'found':>7}"
    print(header)
    print("-" * len(header))
    for row in results:
        for method_name in METHODS:
            m = row[method_name]
            acc = m["score"].get("accuracy")
            acc_str = "-" if acc is None else f"{acc:.2f}"
            tokens = m["input_tokens"] + m["output_tokens"]
            print(f"{row['organism'][:31]:<32}{method_name:<12}{acc_str:>10}{tokens:>9}"
                 f"{str(m['found']):>7}")
    for method_name in METHODS:
        scored = [row[method_name]["score"].get("accuracy") for row in results]
        scored = [s for s in scored if s is not None]
        if scored:
            print(f"\n{method_name}: mean accuracy {sum(scored) / len(scored):.2f} over "
                 f"{len(scored)} paper(s), "
                 f"{sum(row[method_name]['input_tokens'] + row[method_name]['output_tokens'] for row in results):,} tokens total")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=N_PAPERS)
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--provider", choices=["openrouter", "anthropic"],
                        default="openrouter" if os.environ.get("OPENROUTER_API_KEY") else "anthropic")
    args = parser.parse_args()

    judge_model = make_model(args.provider, JUDGE_MODEL_IDS[args.provider])
    results = find_and_evaluate_papers(judge_model, n=args.n, out_path=Path(args.out))

    print(f"\nwrote {args.out}\n")
    print_summary(results)


if __name__ == "__main__":
    main()
