"""Bar-chart plotting for GenoScape's evaluation outputs, kept separate from eval.py
so any already-computed results JSON can be re-plotted on its own -- no model calls,
no re-running anything. Two input shapes are understood:

  --eval-results   eval.py's per-condition comparison (data/eval_results.json): each
                   row is one strain with one sub-object per condition (text, tokens,
                   score). Produces tokens.png and accuracy.png.
  --media-results  save_carbon_source_media_records's raw per-strain output (e.g.
                   data/strain_media_results.json): each row is one strain with a
                   carbon_sources list, each entry carrying provenance about whether
                   its reference's text was fetched and whether a medium was actually
                   extracted from it. Produces media_text_resolved.png and
                   media_extracted.png. This format doesn't record which model produced
                   it, so pass --media-label (e.g. "Sonnet") to show it in the titles.

Add --judge-media-accuracy (with --eval-results and --media-results both given) to also
LLM-judge --media-results' carbon_sources growth calls against --ground-truth, the same
way eval.py judges its own conditions' free-text answers, and fold the result into
accuracy.png as one more bar labeled --media-label -- this makes a live judge-model call.

    python -m examples.plot_results --eval-results data/eval_results.json
    python -m examples.plot_results --media-results data/strain_media_results.json --media-label Sonnet
    python -m examples.plot_results --eval-results data/eval_results.json \\
        --media-results data/strain_media_results.json --media-label "Claude (sonnet)" \\
        --judge-media-accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

from examples.judge import judge_accuracy

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "data" / "eval_plots"

# Categorical slots 1-8 (blue, orange, aqua, yellow, magenta, green, violet, red),
# validated for adjacent-pair CVD/normal-vision separation -- see the dataviz skill's
# palette.md. Fixed order -- callers take positions from the front, never re-sorted or
# re-picked per run.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

CONDITION_COLOR = {"pipeline_haiku": PALETTE[0], "pipeline_sonnet": PALETTE[1],
                   "bare_haiku": PALETTE[2], "bare_sonnet": PALETTE[3]}
CONDITION_LABEL = {"pipeline_haiku": "Genoscape\n(haiku)", "pipeline_sonnet": "Genoscape\n(sonnet)",
                   "bare_haiku": "LLM alone\n(haiku)", "bare_sonnet": "LLM alone\n(sonnet)"}


def _bar_chart(keys: list[str], values: dict[str, float], colors: list[str], labels: list[str],
               title: str, ylabel: str, out_path: Path, value_fmt,
               ylim: tuple[float, float] | None = None) -> None:
    """One bar per key, value labeled at the tip -- no legend needed since the x-axis
    labels already name each bar (a single metric, not several overlapping series)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(keys)), 4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    xs = list(range(len(keys)))
    heights = [values[k] for k in keys]
    ax.bar(xs, heights, width=0.5, color=colors, zorder=3)
    for x, h in zip(xs, heights):
        ax.text(x, h, value_fmt(h), ha="center", va="bottom", fontsize=10, color="#0b0b0b", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color="#52514e")
    ax.set_ylabel(ylabel, fontsize=10, color="#52514e")
    ax.set_title(title, fontsize=12, color="#0b0b0b", pad=12)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:g}")

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e4df", linewidth=1, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e5e4df")
    ax.tick_params(left=False, bottom=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------- eval.py's conditions


def plot_eval_results(results: list[dict], conditions: list[str], out_dir: Path = PLOTS_DIR,
                      extra_accuracy: tuple[str, float] | None = None) -> None:
    """Two bar charts: total tokens burned across all strains (what you actually pay
    for), and average accuracy across all strains. A strain whose judge call didn't
    return parseable JSON (score["accuracy"] is None -- check score["raw_reply"] for
    why) counts as 0 in the average, same as a ground-truth carbon source the answer
    never mentioned -- an unscoreable answer is not evidence of a correct one.

    extra_accuracy, if given, is (label, accuracy) for one more bar on accuracy.png
    only -- e.g. a separately run pipeline's judged accuracy, which has no comparable
    token count to add to tokens.png."""
    total_tokens, avg_accuracy = {}, {}
    for cond in conditions:
        rows = [row[cond] for row in results if cond in row]
        total_tokens[cond] = sum(r["input_tokens"] + r["output_tokens"] for r in rows)
        accs = [r.get("score", {}).get("accuracy") or 0.0 for r in rows]
        avg_accuracy[cond] = sum(accs) / len(accs) if accs else 0.0

    colors = [CONDITION_COLOR[c] for c in conditions]
    labels = [CONDITION_LABEL[c] for c in conditions]
    _bar_chart(conditions, total_tokens, colors, labels, "Total tokens burned", "tokens",
              out_dir / "tokens.png", value_fmt=lambda v: f"{v:,.0f}")

    acc_keys, acc_colors, acc_labels = list(conditions), list(colors), list(labels)
    if extra_accuracy is not None:
        extra_label, extra_value = extra_accuracy
        acc_keys.append(extra_label)
        avg_accuracy[extra_label] = extra_value
        acc_colors.append(PALETTE[len(conditions) % len(PALETTE)])
        acc_labels.append(extra_label)
    _bar_chart(acc_keys, avg_accuracy, acc_colors, acc_labels, "Average accuracy (judged by Opus)",
              "accuracy", out_dir / "accuracy.png", value_fmt=lambda v: f"{v:.2f}", ylim=(0, 1.1))


# ----------------------------------------------------------- media-extraction results


def _media_strain_rates(media_results: list[dict]) -> dict[str, dict[str, float]]:
    """Per strain, out of all its carbon-source entries: the fraction whose reference
    actually had fetchable text (provenance.text_source != "none"), and the fraction
    where a medium composition was actually extracted from it (full_media non-empty).
    The second can only be <= the first -- you can't extract a medium from text you
    never got."""
    rates = {}
    for strain in media_results:
        entries = strain.get("carbon_sources") or []
        n = len(entries) or 1
        text_resolved = sum(1 for e in entries
                            if (e.get("provenance") or {}).get("text_source", "none") != "none")
        media_extracted = sum(1 for e in entries if e.get("full_media"))
        rates[strain["strain_name"]] = {"text_resolved": text_resolved / n,
                                        "media_extracted": media_extracted / n}
    return rates


def _media_results_as_answer_text(strain_media_entry: dict) -> str:
    """Serialize one strain's carbon_sources growth calls into the same free-text
    shape judge_accuracy expects from a model's answer, so the same judge can score
    a raw media-results file's structured growth predictions."""
    lines = [f"{c['carbon_source']}: {'grows' if c.get('growth') else 'does not grow'}"
             for c in strain_media_entry.get("carbon_sources") or []]
    return "\n".join(lines)


def judge_media_growth_accuracy(media_results: list[dict], ground_truth: list[dict],
                                judge_model) -> dict[str, float]:
    """Score each strain in media_results against ground_truth's carbon-source list,
    matched by bacdive_id, using the same LLM-as-judge eval.py uses for its own
    conditions -- just built from media_results' structured carbon_sources instead of
    a model's free-text answer. A judge call that doesn't return parseable JSON scores
    0 for that strain, same convention as plot_eval_results' averaging."""
    gt_by_id = {g["bacdive_id"]: g for g in ground_truth}
    scores = {}
    for strain in media_results:
        gt = gt_by_id.get(strain.get("bacdive_id"))
        if gt is None:
            continue
        answer_text = _media_results_as_answer_text(strain)
        score = judge_accuracy(judge_model, gt["carbon_sources"], answer_text)
        scores[strain["strain_name"]] = score["accuracy"] or 0.0
    return scores


def plot_media_results(media_results: list[dict], model_label: str, out_dir: Path = PLOTS_DIR) -> None:
    """Two bar charts, one per strain in media_results: the fraction of its carbon-source
    entries whose cited reference actually yielded fetchable text, and the fraction where
    a growth-medium composition was actually extracted from that text. model_label names
    whichever model produced media_results in the chart titles, since the file itself
    doesn't record that."""
    rates = _media_strain_rates(media_results)
    strains = list(rates)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(strains))]
    labels = ["\n".join(textwrap.wrap(s, 14)) for s in strains]

    _bar_chart(strains, {s: rates[s]["text_resolved"] for s in strains}, colors, labels,
              f"Reference text resolved ({model_label})", "fraction of carbon sources",
              out_dir / "media_text_resolved.png", value_fmt=lambda v: f"{v:.0%}", ylim=(0, 1.1))
    _bar_chart(strains, {s: rates[s]["media_extracted"] for s in strains}, colors, labels,
              f"Growth medium extracted ({model_label})", "fraction of carbon sources",
              out_dir / "media_extracted.png", value_fmt=lambda v: f"{v:.0%}", ylim=(0, 1.1))


# --------------------------------------------------------------------------- CLI


def _make_judge_model(provider: str):
    from examples.Genoscape import make_model
    opus_ids = {"anthropic": "claude-opus-5", "openrouter": "anthropic/claude-opus-5"}
    return make_model(provider, opus_ids[provider])


def main() -> None:
    from examples.Genoscape import load_env
    load_env()  # before the parser's --provider default reads os.environ
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval-results", help="path to an eval.py-style results JSON")
    parser.add_argument("--conditions", help="comma-separated subset of conditions to plot "
                                              "from --eval-results (default: whatever it contains)")
    parser.add_argument("--media-results", help="path to a save_carbon_source_media_records-style "
                                                "results JSON")
    parser.add_argument("--media-label", default="model unspecified",
                        help="model that produced --media-results, e.g. Sonnet -- shown in its chart titles")
    parser.add_argument("--ground-truth", default=str(ROOT / "data" / "ground_truth.json"),
                        help="ground truth to judge --media-results' growth calls against "
                             "(only used with --judge-media-accuracy)")
    parser.add_argument("--judge-media-accuracy", action="store_true",
                        help="LLM-judge --media-results' carbon-source growth calls against "
                             "--ground-truth (makes a live judge-model call) and fold the result "
                             "into accuracy.png as one more bar, labeled --media-label -- requires "
                             "--eval-results and --media-results both")
    parser.add_argument("--provider", choices=["openrouter", "anthropic"],
                        default="openrouter" if os.environ.get("OPENROUTER_API_KEY") else "anthropic",
                        help="provider for the judge model used by --judge-media-accuracy")
    parser.add_argument("--out-dir", default=str(PLOTS_DIR))
    args = parser.parse_args()
    if not args.eval_results and not args.media_results:
        parser.error("pass --eval-results and/or --media-results")
    if args.judge_media_accuracy and not (args.eval_results and args.media_results):
        parser.error("--judge-media-accuracy requires both --eval-results and --media-results")
    out_dir = Path(args.out_dir)

    media_results = None
    if args.media_results:
        media_results = json.loads(Path(args.media_results).read_text(encoding="utf-8"))

    if args.eval_results:
        results = json.loads(Path(args.eval_results).read_text(encoding="utf-8"))
        if args.conditions:
            conditions = args.conditions.split(",")
        else:
            present = {k for row in results for k in row if k not in ("strain_name", "bacdive_id")}
            conditions = [c for c in CONDITION_COLOR if c in present]

        extra_accuracy = None
        if args.judge_media_accuracy:
            ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
            judge_model = _make_judge_model(args.provider)
            scores = judge_media_growth_accuracy(media_results, ground_truth, judge_model)
            avg = sum(scores.values()) / len(scores) if scores else 0.0
            print("media-results judged accuracy per strain:", scores)
            extra_accuracy = (args.media_label, avg)

        plot_eval_results(results, conditions, out_dir, extra_accuracy=extra_accuracy)

    if media_results is not None:
        plot_media_results(media_results, args.media_label, out_dir)


if __name__ == "__main__":
    main()
