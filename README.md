# GenoScape: An Agentic Pipeline for Mining Genotype × Environment → Growth-Phenotype Data from NCBI, BacDive and the Literature

GenoScape takes the name of a prokaryote (e.g. "Escherichia coli K-12") and
produces a set of JSON records pairing its genome, a carbon source, and
whether it grows on it — with the exact growth medium pulled from the paper
behind that result, where available.

<img src="docs/pipeline.png" width="900" alt="GenoScape pipeline">


## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Add your keys

Create a file named `.env` in the project root with:

```
OPENROUTER_API_KEY=...
```

(or `ANTHROPIC_API_KEY=...` instead, if you use Anthropic directly)

Optional, but recommended if you'll run this a lot:

```
NCBI_API_KEY=...
UNPAYWALL_EMAIL=you@email.com
```

## 3. Run it

```bash
python -m examples.Genoscape --prokaryote_name "Acetobacter cerevisiae"
```

Swap in any organism name you want. It will take a little while — it's
looking the organism up, checking its known growth data, and reading papers.

## 4. Get your results

Each organism gets one file under `data/strain_media/`, named after the
strain and its NCBI ID, e.g. `data/strain_media/acetobacter_cerevisiae_178900.json`.
It lists every carbon source tested, with the growth medium pulled from the
paper for each one:

```json
{
  "strain_name": "Acetobacter cerevisiae",
  "ncbi_id": "178900",
  "strain_id": 12,
  "genome": "GCA_001580535",
  "carbon_sources": [
    {
      "carbon_source": "glycerol",
      "growth": true,
      "full_media": [
        {"name": "GYC medium", "ingredients": [{"name": "glucose", "amount": "50 g/l"}],
         "temperature": "28°C", "ph": "6.8", "notes": null}
      ],
      "provenance": {"reference_title": "...", "doi": "..."}
    },
    {"carbon_source": "ethanol", "growth": false, "full_media": [], "provenance": {"...": "..."}}
  ]
}
```

`full_media` holds whatever a model found by reading the actual paper behind
that result. It's left empty when the paper wasn't readable or didn't
describe a medium — it's never guessed. Reading papers uses the same key you
set up in step 2.

## 5. Evaluate it (optional)

Compare the full pipeline against a bare LLM with no tools, on a small set of
strains with known answers (`data/ground_truth.json`):

```bash
python -m examples.eval
```

This runs 4 conditions (the pipeline and a bare LLM, each with Haiku and
Sonnet), scores each answer with an LLM judge, and writes:

- `data/eval_results.json` — every answer and its judged accuracy
- `data/eval_plots/tokens.png` and `accuracy.png` — bar charts comparing all 4

To re-draw those charts later without re-running anything (e.g. after editing
`data/eval_results.json`, or to plot a subset of conditions):

```bash
python -m examples.plot_results --eval-results data/eval_results.json
```

## Licence

MIT. See [LICENSE](LICENSE).
