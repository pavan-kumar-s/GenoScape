"""Build a small evaluation ground truth: a handful of random BacDive strains that
have carbon-source growth data, with their key facts -- everything except the growth
medium itself, which is what the pipeline (save_carbon_source_media_records) is
actually being evaluated on.

    python -m examples.build_ground_truth

Writes data/ground_truth.json.
"""

import json
import random
from pathlib import Path

from examples.my_tools import (ToolError, _best_genome_accession, _fetch_bacdive_record,
                               _ncbi_taxids, get_bacdive_carbon_source_growth)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "ground_truth.json"
N_STRAINS = 5
SEED = 42  # fixed so the "random" picks are reproducible
ID_RANGE = (1, 179500)  # BacDive-IDs are sparse above this; see examples/my_tools.py


def _strain_name(record: dict) -> str:
    """Species name plus strain designation, e.g. 'Escherichia coli K-12'."""
    tax = record.get("Name and taxonomic classification") or {}
    tax = tax[0] if isinstance(tax, list) else tax
    species = (tax or {}).get("species") or ""
    designation = (tax or {}).get("strain designation")
    return f"{species} {designation}".strip() if designation else species


def _best_ncbi_id(record: dict) -> str | None:
    """Prefer a strain-level NCBI tax id over the species-level default."""
    ids = _ncbi_taxids(record)
    if not ids:
        return None
    best = next((i for i in ids if str(i.get("Matching level")).lower() == "strain"), ids[0])
    taxid = best.get("NCBI tax id")
    return str(taxid) if taxid is not None else None


def pick_ground_truth(n: int = N_STRAINS, seed: int = SEED,
                      id_range: tuple[int, int] = ID_RANGE) -> list[dict]:
    """Randomly sample BacDive-IDs (reproducibly, given `seed`) until `n` strains with
    carbon-source/growth utilization data are found. Each result has the facts the
    pipeline should be able to reproduce on its own -- NOT the growth medium, since
    that's the one thing it has to go dig up from the literature."""
    rng = random.Random(seed)
    lo, hi = id_range
    found, tried = [], set()
    while len(found) < n:
        bid = rng.randint(lo, hi)
        if bid in tried:
            continue
        tried.add(bid)
        try:
            tests = get_bacdive_carbon_source_growth(bid)["tests"]
        except ToolError:
            continue  # invalid id, or no carbon-source data -- try another
        record = _fetch_bacdive_record(bid)
        ncbi_id = _best_ncbi_id(record)
        genome = _best_genome_accession(record, ncbi_id)
        found.append({
            "bacdive_id": bid,
            "strain_name": _strain_name(record),
            "ncbi_id": ncbi_id,
            "genome": genome,
            "genome_link": f"https://www.ncbi.nlm.nih.gov/datasets/genome/{genome}/" if genome else None,
            "carbon_sources": [{"carbon_source": t["metabolite"], "growth": t["growth"],
                               "ref": t["ref"]} for t in tests],
        })
    return found


if __name__ == "__main__":
    ground_truth = pick_ground_truth()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(ground_truth)} strains to {OUT_PATH}")
    for s in ground_truth:
        print(f"  {s['bacdive_id']}: {s['strain_name']} ({len(s['carbon_sources'])} carbon sources)")
