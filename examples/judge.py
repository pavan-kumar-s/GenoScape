"""Shared LLM-as-judge scoring. Two judges live here:

  judge_accuracy          -- given a ground-truth strain's carbon-source/growth pairs
                             and a candidate answer (any free text), scores one
                             accuracy number. Used by examples/eval.py (judging
                             free-text pipeline/bare-LLM answers) and
                             examples/plot_results.py (judging a raw media-results
                             file's structured carbon_sources, serialized to the same
                             free-text shape).
  judge_media_extraction  -- no ground truth at all: given the complete paper text,
                             the instructions the extraction model was given, and its
                             output, judges whether that output faithfully reflects
                             what the paper itself says. Used by examples/eval_media.py.
  judge_paper_has_media   -- does this paper state a medium's composition at all? Used
                             by examples/eval_media.py to choose which papers are worth
                             evaluating, independently of what any extraction method
                             claims to have found.
"""

from __future__ import annotations

import json
import re

JUDGE_SYSTEM = (
    "You are grading an answer about bacterial growth on different carbon sources. "
    "You are given a ground-truth list of carbon sources with the correct growth result "
    "for each, and a candidate answer written in free text (it may or may not be JSON, "
    "and may use different wording or field names). For EVERY carbon source in the "
    "ground-truth list -- and ONLY those; ignore any other carbon source the candidate "
    "answer mentions that is not in the ground-truth list -- decide whether the "
    "candidate answer's growth call for it matches the ground truth. If the candidate "
    "answer does not mention a ground-truth carbon source at all (by name or a clear "
    "synonym), that counts as incorrect for it. Respond with ONLY a JSON object, no "
    "other text and no markdown fences: {\"accuracy\": <float between 0 and 1>}, equal "
    "to the fraction of ground-truth carbon sources the candidate answer got right."
)


def judge_accuracy(judge_model, ground_truth_carbon_sources: list[dict], answer_text: str) -> dict:
    """LLM-as-judge: read the raw answer text directly (no parsing of it by us at all)
    and score it against the ground truth's carbon-source/growth pairs, ignoring any
    carbon source not in the ground truth. 700 max_tokens because the judge has to name
    and cross-check every ground-truth carbon source (up to ~70 for some strains here)
    before it can emit the JSON -- 200 was too tight and silently truncated the judge's
    own reply on the larger comparisons. Returns {"accuracy": None, "raw_reply": ...}
    (not a bare None) when the reply still isn't parseable, so a bad judge call is
    diagnosable afterwards instead of just vanishing -- one bad call shouldn't kill a
    whole run, but it shouldn't be silently unexplainable either."""
    gt_lines = "\n".join(f"- {c['carbon_source']}: {'grows' if c['growth'] else 'does not grow'}"
                         for c in ground_truth_carbon_sources)
    prompt = f"Ground truth carbon sources and growth results:\n{gt_lines}\n\nCandidate answer:\n{answer_text}"
    response = judge_model.complete(JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
                                     tools=[], max_tokens=700)
    raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", response.text.strip())
    try:
        return {"accuracy": float(json.loads(raw)["accuracy"]), "raw_reply": None}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"accuracy": None, "raw_reply": response.text}


JUDGE_MEDIA_SYSTEM = (
    "You are grading whether an automated growth-medium extraction correctly followed its "
    "own instructions. You will be given: the strain the paper concerns, the complete text "
    "of the scientific paper, the exact instructions the extraction model was given, and the "
    "JSON that model actually produced. Judge the output ONLY against what the paper's text "
    "itself actually states -- never your own outside knowledge of this organism or medium. "
    "Score it on: (1) did it correctly report whether the paper states a medium's composition "
    "at all; (2) if it reported one, does every ingredient/amount/temperature/pH it gives "
    "match what the paper actually says, with nothing invented and nothing the paper states "
    "left out. Respond with ONLY a JSON object, no other text and no markdown fences: "
    "{\"accuracy\": <float between 0 and 1>, \"reason\": \"<one sentence, citing specifics>\"}."
)


PAPER_HAS_MEDIA_SYSTEM = (
    "You decide whether a scientific paper states the COMPOSITION of a culture medium -- "
    "the actual ingredients a medium is made of, with or without amounts (e.g. 'per litre: "
    "5 g peptone, 3 g yeast extract'). A medium merely NAMED (e.g. 'grown on ISP 2 agar', "
    "'marine broth 2216'), or named alongside growth observations or test results, is NOT "
    "a composition. Respond with ONLY a JSON object, no other text and no markdown fences: "
    "{\"has_media\": <true|false>, \"reason\": \"<one sentence, citing specifics>\"}."
)


def judge_paper_has_media(judge_model, strain_name: str, paper_text: str) -> dict:
    """Does this paper actually state a growth medium's composition? Used to pick which
    papers are worth evaluating extraction on, INDEPENDENTLY of what any extraction
    method reports.

    Gating on an extraction method's own "found" instead would bias the whole eval: it
    lets that method's false positives in (a datasheet naming 'Agar suter without
    tyrosine' with no ingredients got accepted that way), and -- worse -- excludes the
    single most valuable test case, a paper that really does contain a recipe the method
    missed. Returns {"has_media": None, "reason": None, "raw_reply": ...} when the reply
    isn't parseable JSON, so a bad call stays diagnosable."""
    prompt = f"Strain the paper concerns: {strain_name}\n\nComplete paper text:\n{paper_text}"
    # 600, not 300: the reason routinely quotes a whole ingredient list as evidence, and a
    # truncated reply parses as has_media=None -- which a caller reads as "no medium here"
    # and silently drops a paper that actually had one.
    response = judge_model.complete(PAPER_HAS_MEDIA_SYSTEM, [{"role": "user", "content": prompt}],
                                     tools=[], max_tokens=600)
    raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", response.text.strip())
    try:
        parsed = json.loads(raw)
        return {"has_media": bool(parsed["has_media"]), "reason": parsed.get("reason"),
                "raw_reply": None}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"has_media": None, "reason": None, "raw_reply": response.text}


def judge_media_extraction(judge_model, strain_name: str, extraction_instructions: str,
                           paper_text: str, extraction_output: dict) -> dict:
    """LLM-as-judge for growth-medium extraction, with no hand-written ground truth: the
    judge reads the complete paper itself (paper_text), the instructions the extracting
    model was given (extraction_instructions -- my_tools.EXTRACTION_SYSTEM), and that
    model's output, and scores how faithfully the output reflects what the paper actually
    says. Returns {"accuracy": None, "reason": None, "raw_reply": ...} (not a bare None)
    when the reply isn't parseable JSON, so a bad judge call stays diagnosable."""
    prompt = (
        f"Strain: {strain_name}\n\n"
        f"Extraction instructions given to the model:\n{extraction_instructions}\n\n"
        f"Complete paper text:\n{paper_text}\n\n"
        f"Model's extracted output:\n{json.dumps(extraction_output, indent=2, ensure_ascii=False)}"
    )
    # 900, not 400: the reason cites specific ingredients and amounts to justify the score,
    # and 400 truncated it mid-list on exactly the papers with the most to say -- the score
    # was already written by then, so the whole verdict was lost to an unparseable reply.
    response = judge_model.complete(JUDGE_MEDIA_SYSTEM, [{"role": "user", "content": prompt}],
                                     tools=[], max_tokens=900)
    raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", response.text.strip())
    try:
        parsed = json.loads(raw)
        return {"accuracy": float(parsed["accuracy"]), "reason": parsed.get("reason"), "raw_reply": None}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"accuracy": None, "reason": None, "raw_reply": response.text}
