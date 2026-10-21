"""Shared LLM-as-judge scoring: given a ground-truth strain's carbon-source/growth
pairs and a candidate answer (any free text), ask a fixed judge model for one
accuracy number. Shared by examples/eval.py (judging free-text pipeline/bare-LLM
answers) and examples/plot_results.py (judging a raw media-results file's structured
carbon_sources list, after serializing it to the same free-text shape)."""

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
