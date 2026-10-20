# GenoScape: An Agentic Pipeline for Mining Genotype × Environment → Growth-Phenotype Data from NCBI, BacDive and the Literature

GenoScape is a tool-using agent that turns a prokaryote's name into structured,
source-linked growth-phenotype data. Given an organism (e.g. *Escherichia coli*
K-12), it resolves the strain across NCBI Taxonomy and BacDive, reads off its
carbon-source utilization tests, tracks down and reads the paper behind each
test, and extracts the exact growth medium reported there — producing records
that pair a **genotype** (strain + genome accession) and an **environment**
(carbon source, growth medium) with an observed **growth phenotype**, each one
traceable back to its NCBI ID, BacDive ID, and literature reference.

The goal (see [SPEC.md](SPEC.md)) is to build the kind of dataset an ML/DL
model would need to predict whether a given microbe can be cultivated in a
given environment — useful to researchers trying to grow organisms observed in
the wild but not yet grown in the lab.

## Why an agent?

A single API call can't answer "what medium grows this strain on this carbon
source, and where did that fact come from?":

- **No single database has the full triad.** NCBI has taxonomy and genomes,
  BacDive has curated phenotype tests, and the actual media recipe is often
  only in the cited paper's methods section — sometimes behind a paywall,
  sometimes in PubMed Central, sometimes nowhere machine-readable at all.
- **Resolution is ambiguous and must be iterative.** "E. coli K-12" needs to
  become a specific NCBI tax ID, then a specific BacDive strain record (BacDive
  pages results by ID, not relevance, so the right strain may not be in the
  first batch returned), before any phenotype data can be pulled.
- **Literature access is a decision tree, not a single call.** Resolve a DOI,
  try PubMed Central's structured full text first, fall back to
  Crossref/Unpaywall/a direct fetch if the article isn't deposited there, and
  be honest when nothing is actually readable (most papers are still
  paywalled).

Each step's result shapes the next call, which is exactly the loop a
tool-using agent is for.

## How it works

GenoScape is built on top of `agentbook/`, a small tool-using agent harness
(a model adapter, a `ToolRegistry` that validates arguments and turns
failures into messages the model can act on, a run loop, and JSONL tracing).
The domain-specific tools live in [examples/my_tools.py](examples/my_tools.py);
the agent entry point is [examples/Genoscape.py](examples/Genoscape.py).

## Tools

| Stage | Tool | What it does |
|---|---|---|
| Resolve | `get_ncbi_strain_names` | Prokaryote name → NCBI Taxonomy ID, rank, and every alternate name on record (synonyms, common names, type-strain deposit numbers). |
| Resolve | `get_bacdive_details` | Taxonomic name (+ optional NCBI tax ID) → matching BacDive record(s): BacDive ID, NCBI mapping, and every section (morphology, culture conditions, physiology, sequence info, literature). |
| Resolve | `pick_top_bacdive_strain` | Chains the two above end-to-end and picks **one** strain (NCBI-matched → type strain → lowest BacDive-ID), so nothing downstream has to disambiguate. |
| Phenotype | `get_bacdive_carbon_source_growth` | One strain's carbon-source utilization tests only: metabolite, growth (+/-), and the reference backing each test. |
| Literature | `get_pubmed_reference_text` | DOI → full text from PubMed Central, when the article is deposited there. |
| Literature | `get_bacdive_reference_text` | DOI/URL → text via Crossref + Unpaywall + a direct fetch, as a fallback when an article isn't in PMC. |
| Extraction | `extract_media_composition_candidates` | Scans fetched text for lines that look like media composition (quantity + unit next to a media keyword) — a keyword filter, not language understanding; candidates to verify, not a guaranteed recipe. |
| Save | `save_carbon_source_media_record` | Runs the literature step for one carbon-source test (PMC first, fallback second), extracts media candidates, and writes a JSON record to `data/strain_media/`. |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root (never commit it):

```bash
OPENROUTER_API_KEY=...        # or ANTHROPIC_API_KEY, for examples/Genoscape.py
NCBI_API_KEY=...              # optional: raises the NCBI rate limit from ~3 to ~10 req/s
UNPAYWALL_EMAIL=you@email.com # optional: lets get_bacdive_reference_text check for an open-access copy
```

Optionally, `pip install pypdf` so `get_bacdive_reference_text` can extract text from a
fetched PDF (without it, a PDF comes back as a placeholder string instead of its text).

Python 3.10+. Run the offline test suite (no API key needed, a few seconds):

```bash
python -m pytest
```

## Running it

```bash
python -m examples.Genoscape --prokaryote_name "Escherichia coli K-12"
```

flags: `--provider` (openrouter/anthropic), `--model`, `--plan` (adds a
plan/reflect loop), `--max-steps`, `--budget` (token cap). Every run is traced
to `traces/live_<time>.jsonl`.

You can also drive the pipeline directly, one tool call at a time, without a
model — useful for testing or scripting a fixed pipeline:

```python
from examples.my_tools import (pick_top_bacdive_strain, get_bacdive_carbon_source_growth,
                               save_carbon_source_media_record)

strain = pick_top_bacdive_strain("Escherichia coli")
tests = get_bacdive_carbon_source_growth(strain["bacdive_id"])["tests"]
for test in tests:
    if test["growth"]:
        save_carbon_source_media_record(
            bacdive_id=strain["bacdive_id"], metabolite=test["metabolite"],
            ref_id=test["ref"], ncbi_id=strain["ncbi_taxid"])
```

Each call writes `data/strain_media/<bacdive_id>_<carbon_source>.json`:

```json
{
  "carbon_source": "glucose",
  "growth": true,
  "genome": "GCA_900699165",
  "ncbi_id": "562",
  "strain_id": 4434,
  "full_media": [],
  "provenance": {"ref_id": 31279, "reference_title": "...", "doi": "...", "text_source": "pubmed_central", "pmcid": "PMC5541158"}
}
```

`full_media` is only ever what was actually found in the fetched paper text —
it's left empty rather than silently backfilled from BacDive's own structured
`culture medium` field, so every record's provenance stays honest.

## Status

Implemented: strain resolution, carbon-source phenotype extraction, a
two-tier literature-fetch strategy (PMC → Crossref/Unpaywall/direct), and
best-effort media-composition extraction with JSON output.

Not yet built, per the original plan in [SPEC.md](SPEC.md): a dedicated
MediaDive API tool (media composition currently comes from BacDive's own
record or a keyword scan of paper text, not a direct MediaDive query), a
Fitness Browser (RB-TnSeq) tool for gene-level perturbation data, and strict
Pydantic schema validation of extracted records (current extraction is a
keyword filter, not a validated structured parse).

## Layout

```
agentbook/            the underlying tool-using agent harness (model adapter,
                       ToolRegistry, run loop, tracing, plan/reflect scaffolding)
examples/
  Genoscape.py         GenoScape's agent entry point
  my_tools.py          the GenoScape tools (table above)
  minimal_loop.py, lyman_alpha_plan.py, failure_trace.py, run_live.py
                       the original dark-matter-registry examples this
                       harness was built for -- unrelated to GenoScape's
                       domain, kept as a second worked example of the harness
data/
  strain_media/        JSON records written by save_carbon_source_media_record
  registry_snapshot/   dark-matter model cards used by the examples above
traces/                JSONL traces of agent runs
tests/                 pytest, offline
SPEC.md                the original project spec and task definition
```

## Licence

The code is under the [MIT licence](LICENSE). `data/registry_snapshot/` comes
from DarkMatterFM-Registry and keeps that repository's own terms; see
`data/registry_snapshot/REGISTRY_COPYRIGHT.md`.
