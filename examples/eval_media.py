"""Compare two ways of parsing culture-medium composition out of a paper's text:

  1. whole_text -- my_tools.extract_media_composition_whole_text: upload the
     entire text to an LLM in one call.
  2. retrieval  -- my_tools.extract_media_composition (the method actually used by
     save_carbon_source_media_records): rank paragraphs by concentration-unit
     density, verify + extract from the best one, retry the next-best if not found.

    python -m examples.eval_media

Runs both methods on one real citation (see CITATION below -- a real species-
description paper, fetched the same way the pipeline would fetch it), reports
token usage for each, and -- once data/media_ground_truth.json has been filled
in by hand -- also scores each method's extracted ingredients against it.
"""

from __future__ import annotations

import json
from pathlib import Path

from examples.Genoscape import load_env
from examples.my_tools import (ToolError, get_pubmed_reference_text,
                               extract_media_composition, extract_media_composition_whole_text)

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = ROOT / "data" / "media_ground_truth.json"

# A real, fully open-access species-description paper -- the kind save_carbon_source_media_records
# resolves a BacDive reference to via get_pubmed_reference_text (PMC full text is the pipeline's
# first choice, tried before Crossref/Unpaywall/direct-fetch) -- with a concrete growth-medium
# recipe (ISP agars, NaCl/pH/temperature ranges, a carbon-source test medium with 0.01% w/v yeast
# extract) that's easy to read and check by eye at the PMC link below.
CITATION = {
    "doi": "10.3389/fmicb.2018.01587",
    "pmcid": "PMC6058180",
    "title": "Streptomyces nigra sp. nov. Is a Novel Actinobacterium Isolated From Mangrove "
             "Soil and Exerts a Potent Antitumor Activity in Vitro",
    "pmc_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6058180/",
}


def fetch_citation_text() -> str:
    """The text both extraction methods run on, fetched the same way the real pipeline does:
    get_pubmed_reference_text pulls this DOI's full text straight from PubMed Central -- no
    Unpaywall/UNPAYWALL_EMAIL needed, since this paper is openly deposited there."""
    result = get_pubmed_reference_text(CITATION["doi"])
    text = result.get("text") or ""
    if not text:
        raise RuntimeError(f"get_pubmed_reference_text({CITATION['doi']!r}) returned no text: "
                           f"{result.get('status')}")
    return text


def _flatten_ingredients(media: list[dict]) -> dict[str, str | None]:
    """{ingredient name (lowercase): amount} across every medium an extraction found."""
    out = {}
    for m in media or []:
        for ing in (m.get("ingredients") or []):
            name = str(ing.get("name", "")).strip().lower()
            if name:
                out[name] = ing.get("amount")
    return out


def score_media(predicted: list[dict], ground_truth: list[dict]) -> dict:
    gt, pred = _flatten_ingredients(ground_truth), _flatten_ingredients(predicted)
    matched = set(gt) & set(pred)
    amount_matches = sum(1 for k in matched if str(pred[k]).strip().lower() == str(gt[k]).strip().lower())
    return {
        "n_ground_truth_ingredients": len(gt), "n_predicted_ingredients": len(pred),
        "n_matched_names": len(matched),
        "ingredient_recall": round(len(matched) / len(gt), 3) if gt else None,
        "amount_exact_match_on_matched": round(amount_matches / len(matched), 3) if matched else None,
    }


def run_method(name: str, fn, text: str) -> dict:
    usage_log: list[dict] = []
    try:
        result = fn(text, usage_log=usage_log)
    except ToolError as e:
        result, usage_log = {"found": False, "media": [], "error": str(e)}, usage_log
    tokens_in = sum(u["input_tokens"] for u in usage_log)
    tokens_out = sum(u["output_tokens"] for u in usage_log)
    return {"method": name, "n_llm_calls": len(usage_log), "input_tokens": tokens_in,
            "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out,
            "found": result.get("found", False), "media": result.get("media", []),
            "attempts": result.get("attempts")}


def main() -> None:
    load_env()
    text = fetch_citation_text()
    print(f"citation: {CITATION['title']}")
    print(f"doi: {CITATION['doi']}  |  fetched text: {len(text)} chars\n")

    outcomes = [
        run_method("whole_text", extract_media_composition_whole_text, text),
        run_method("retrieval", extract_media_composition, text),
    ]

    ground_truth = None
    if GROUND_TRUTH_PATH.exists():
        gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        if gt.get("media"):
            ground_truth = gt["media"]

    for outcome in outcomes:
        if ground_truth is not None:
            outcome["score"] = score_media(outcome["media"], ground_truth)

    print(f"{'method':<12}{'llm_calls':>11}{'in_tok':>9}{'out_tok':>9}{'total_tok':>11}{'found':>8}", end="")
    print(f"{'recall':>9}{'amt_match':>11}" if ground_truth is not None else "")
    for o in outcomes:
        row = f"{o['method']:<12}{o['n_llm_calls']:>11}{o['input_tokens']:>9}{o['output_tokens']:>9}" \
              f"{o['total_tokens']:>11}{str(o['found']):>8}"
        if ground_truth is not None:
            s = o["score"]
            recall_str = "-" if s["ingredient_recall"] is None else format(s["ingredient_recall"], ".2f")
            amt_str = ("-" if s["amount_exact_match_on_matched"] is None
                      else format(s["amount_exact_match_on_matched"], ".2f"))
            row += f"{recall_str:>9}{amt_str:>11}"
        print(row)

    print()
    for o in outcomes:
        print(f"--- {o['method']} extracted ---")
        print(json.dumps(o["media"], indent=2))
        print()

    if ground_truth is None:
        print(f"No ground truth yet -- fill in {GROUND_TRUTH_PATH} (scaffold already "
              "written with the real excerpt) and re-run to see accuracy scores too.")


if __name__ == "__main__":
    main()
