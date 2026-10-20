# GenoScape: An Agentic Pipeline for Mining Genotype × Environment → Growth-Phenotype Data from NCBI, BacDive and the Literature

GenoScape takes the name of a prokaryote (e.g. "Escherichia coli K-12") and
produces a set of JSON records pairing its genome, a carbon source, and
whether it grows on it — with the exact growth medium pulled from the paper
behind that result, where available.

## 1. Install

```bash
pip install -r requirements.txt
pip install pypdf   # optional, lets it read PDFs when a paper is only available as one
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
python -m examples.Genoscape --prokaryote_name "Escherichia coli K-12"
```

Swap in any organism name you want. It will take a little while — it's
looking the organism up, checking its known growth data, and reading papers.

## 4. Get your results

Each result is saved as its own file under `data/strain_media/`, named after
the strain and carbon source, e.g. `data/strain_media/4434_glucose.json`:

```json
{
  "carbon_source": "glucose",
  "growth": true,
  "genome": "GCA_900699165",
  "ncbi_id": "562",
  "strain_id": 4434,
  "full_media": [],
  "provenance": {"reference_title": "...", "doi": "..."}
}
```

`full_media` lists the growth-medium details found in the paper. It's left
empty when the paper didn't have that information — it's never guessed.

## Licence

MIT. See [LICENSE](LICENSE).
