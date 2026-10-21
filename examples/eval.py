"""Compare four ways of answering the GenoScape task, against the ground truth in
data/ground_truth.json (see examples/build_ground_truth.py):

  1. pipeline_haiku  -- the full tool-using agent, Haiku model
  2. pipeline_sonnet -- the full tool-using agent, Sonnet model
  3. bare_haiku      -- Haiku given the identical task, no tools at all
  4. bare_sonnet     -- Sonnet given the identical task, no tools at all

    python -m examples.eval
    python -m examples.eval --limit 1 --conditions bare_haiku,bare_sonnet

Scored by an LLM judge (fixed at Opus, regardless of which model produced the
answer, so judging criteria stay consistent across conditions) that reads the raw
answer text directly -- no regex/JSON parsing of the answer at all, since a model's
output format is unpredictable. The judge is given only the ground truth's carbon
sources and growth results, and for EVERY one of those (ignoring any other carbon
source the answer might mention) decides whether the answer's growth call matches;
not mentioning a ground-truth carbon source at all counts as wrong for it. Returns
one accuracy number: correct / (all ground-truth carbon sources).

Writes one trace per pipeline run to traces/, and the full results (including the
free-text answers and the judge's accuracy) to data/eval_results.json.

Plotting lives separately in examples/plot_results.py (so results can be re-plotted
without re-running anything); this script still calls it after a run unless --no-plots.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from agentbook import ToolRegistry, Tracer, run_agent
from agentbook.guardrails import check_cited_ids, repeated_calls, tool_outputs_from_trace
from agentbook.provenance import check_numbers
from agentbook.tracing import last_run, load_trace, summarize_trace
from examples.Genoscape import DEFAULT_TASK, load_env, make_model
from examples.judge import judge_accuracy
from examples.my_tools import MY_TOOLS
from examples.plot_results import plot_eval_results

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth.json"
RESULTS_PATH = ROOT / "data" / "eval_results.json"
PLOTS_DIR = ROOT / "data" / "eval_plots"

MODEL_IDS = {
    ("haiku", "anthropic"): "claude-haiku-4.5", ("haiku", "openrouter"): "anthropic/claude-haiku-4.5",
    ("sonnet", "anthropic"): "claude-sonnet-5", ("sonnet", "openrouter"): "anthropic/claude-sonnet-5",
    ("opus", "anthropic"): "claude-opus-5", ("opus", "openrouter"): "anthropic/claude-opus-5",
}
CONDITIONS = ["pipeline_haiku", "pipeline_sonnet", "bare_haiku", "bare_sonnet"]
JUDGE_SIZE = "opus"


# ----------------------------------------------------------------- running conditions


def run_pipeline(model, task: str, trace_path: Path) -> dict:
    tracer = Tracer(trace_path)
    result = run_agent(model, ToolRegistry(MY_TOOLS), task, max_steps=50,
                       max_tokens_total=200_000, tracer=tracer)
    events = last_run(load_trace(trace_path))
    evidence = tool_outputs_from_trace(events)
    ids = check_cited_ids(result.text, evidence)
    numbers = check_numbers(result.text, evidence)
    return {
        "text": result.text, "steps": result.steps, "stop_reason": result.stop_reason,
        "input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens,
        "trace": str(trace_path), "trace_summary": summarize_trace(events),
        "ids_cited": len(ids.cited), "ids_unsupported": ids.unsupported,
        "numbers_cited": len(numbers.claims), "numbers_unsupported": len(numbers.unsupported),
        "repeated_calls": repeated_calls(result.messages),
    }


def run_bare(model, task: str) -> dict:
    response = model.complete("", [{"role": "user", "content": task}], tools=[], max_tokens=3000)
    return {
        "text": response.text, "steps": 1, "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
    }


def evaluate(ground_truth_entry: dict, outcome: dict, judge_model) -> dict:
    score = judge_accuracy(judge_model, ground_truth_entry["carbon_sources"], outcome["text"])
    return {**outcome, "score": score}


# --------------------------------------------------------------------------- CLI


def print_summary(results: list[dict], conditions: list[str]) -> None:
    header = f"{'strain':<35}{'condition':<16}{'accuracy':>10}{'tokens':>10}{'steps':>7}"
    print(header)
    print("-" * len(header))
    for row in results:
        for cond in conditions:
            c = row[cond]
            accuracy = c.get("score", {}).get("accuracy")
            tokens = c["input_tokens"] + c["output_tokens"]
            acc_str = "-" if accuracy is None else f"{accuracy:.2f}"
            print(f"{row['strain_name'][:34]:<35}{cond:<16}{acc_str:>10}{tokens:>10}{c['steps']:>7}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--out", default=str(RESULTS_PATH))
    parser.add_argument("--limit", type=int, help="only the first N ground-truth strains")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help=f"comma-separated subset of {CONDITIONS}")
    parser.add_argument("--provider", choices=["openrouter", "anthropic"],
                        default="openrouter" if os.environ.get("OPENROUTER_API_KEY") else "anthropic")
    parser.add_argument("--force", action="store_true",
                        help="re-run every (strain, condition) even if --out already has a result for it")
    parser.add_argument("--no-plots", action="store_true", help="skip writing the bar charts")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    out_path = Path(args.out)

    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    if args.limit:
        ground_truth = ground_truth[:args.limit]

    cached = {}
    if out_path.exists() and not args.force:
        for row in json.loads(out_path.read_text(encoding="utf-8")):
            cached[row["bacdive_id"]] = row

    needed_sizes = {("sonnet" if "sonnet" in cond else "haiku") for cond in conditions}
    models = {size: make_model(args.provider, MODEL_IDS[(size, args.provider)]) for size in needed_sizes}
    judge_model = make_model(args.provider, MODEL_IDS[(JUDGE_SIZE, args.provider)])

    results = []
    for gt in ground_truth:
        print(f"\n=== {gt['strain_name']} (bacdive_id {gt['bacdive_id']}) ===")
        task = DEFAULT_TASK.format(prokaryote_name=gt["strain_name"])
        cached_row = cached.get(gt["bacdive_id"], {})
        # Seed from cached_row first: a --conditions subset must not drop previously
        # computed conditions that aren't in this run's subset -- only overwrite the
        # ones actually requested below.
        row = {**cached_row, "strain_name": gt["strain_name"], "bacdive_id": gt["bacdive_id"]}
        for cond in conditions:
            if cond in cached_row and "accuracy" in cached_row[cond].get("score", {}):
                print(f"  {cond}: already have a scored result, skipping")
                row[cond] = cached_row[cond]
                continue
            if cond in cached_row and "text" in cached_row[cond]:
                # The answer was already generated (expensive), but its score predates
                # the current scoring method (e.g. the LLM-judge rewrite) -- reuse the
                # answer, just re-score it, instead of re-running the model for it.
                print(f"  {cond}: have the answer, but an outdated score -- re-judging only")
                row[cond] = evaluate(gt, cached_row[cond], judge_model)
                continue
            size = "sonnet" if "sonnet" in cond else "haiku"
            print(f"  running {cond}...")
            if cond.startswith("pipeline_"):
                trace_path = ROOT / "traces" / f"eval_{cond}_{gt['bacdive_id']}.jsonl"
                outcome = run_pipeline(models[size], task, trace_path)
            else:
                outcome = run_bare(models[size], task)
            row[cond] = evaluate(gt, outcome, judge_model)
        results.append(row)
        # Save after every strain, not just at the end, so an interrupted run still
        # leaves the already-done (strain, condition) results in place to reuse next time.
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote {out_path}\n")
    print_summary(results, conditions)
    if not args.no_plots:
        plot_eval_results(results, conditions)


if __name__ == "__main__":
    main()
