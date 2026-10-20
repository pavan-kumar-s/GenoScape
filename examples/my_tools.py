"""Your own tools. run_live.py adds everything in MY_TOOLS to the registry tools.

A tool is a name, a description the model reads, a JSON schema for its
arguments, and a Python function. Raise ToolError with a message the model
can act on when something goes wrong; it goes back to the model as a result.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import os

from agentbook import Tool, ToolError  # noqa: F401  (ToolError is for your tools)
from agentbook.registry import STATUS, default_registry  # noqa: F401

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_get(endpoint: str, **params) -> str:
    """GET one E-utilities endpoint, raw text. Raises ToolError on any failure."""
    params = {"tool": "genoscape_agentbook", **params}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    url = f"{NCBI_EUTILS}/{endpoint}.fcgi?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise ToolError("NCBI rate-limited this request (HTTP 429). Wait a moment and "
                            "retry with fewer calls, or set an NCBI_API_KEY.") from e
        raise ToolError(f"NCBI {endpoint} returned HTTP {e.code}. Check the organism name "
                        "and retry.") from e
    except urllib.error.URLError as e:
        raise ToolError(f"Could not reach NCBI E-utilities ({e.reason}). Check network "
                        "access and retry.") from e


def _names(other_names: ET.Element | None, tag: str) -> list[str]:
    if other_names is None:
        return []
    return [e.text for e in other_names.findall(tag) if e.text]


def get_ncbi_strain_names(prokaryote_name: str, max_results: int = 5) -> dict:
    """Resolve a prokaryote name to NCBI Taxonomy entries, each with every name on
    record for it (synonyms, common names, equivalent names, GenBank 'includes'
    entries, and type-strain deposit numbers) plus its NCBI taxonomy ID."""
    search = json.loads(_ncbi_get("esearch", db="taxonomy", term=prokaryote_name,
                                  retmode="json", retmax=max_results))
    ids = (search.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        raise ToolError(f"No NCBI Taxonomy entry found for '{prokaryote_name}'. Try a shorter "
                        "or differently spelled name (e.g. drop a strain suffix).")
    xml_text = _ncbi_get("efetch", db="taxonomy", id=",".join(ids), retmode="xml")
    root = ET.fromstring(xml_text)
    taxa = []
    for taxon in root.findall("Taxon"):
        other = taxon.find("OtherNames")
        type_material = [n.findtext("DispName") for n in (other.findall("Name") if other is not None else [])
                         if n.findtext("ClassCDE") == "type material" and n.findtext("DispName")]
        taxa.append({
            "ncbi_taxid": taxon.findtext("TaxId"),
            "scientific_name": taxon.findtext("ScientificName"),
            "rank": taxon.findtext("Rank"),
            "synonyms": _names(other, "Synonym"),
            "common_names": _names(other, "CommonName"),
            "genbank_common_names": _names(other, "GenbankCommonName"),
            "equivalent_names": _names(other, "EquivalentName"),
            "includes": _names(other, "Includes"),
            "type_material": type_material,
        })
    return {
        "query": prokaryote_name,
        "total_found": len(ids),
        "taxa": taxa,
        "source": "NCBI Taxonomy E-utilities (esearch + efetch)",
    }


MY_TOOLS: list[Tool] = [
    Tool(
        "get_ncbi_strain_names",
        "Look up a prokaryote or strain name in NCBI Taxonomy. Returns, for each matching "
        "taxon (up to `max_results`), its NCBI taxonomy ID, rank, and every alternate name on "
        "record: synonyms, common names, equivalent names, GenBank 'includes' entries, and "
        "type-strain deposit numbers (e.g. ATCC/DSM numbers). Use this first to pin an "
        "informally written organism name (e.g. 'E. coli K12') to its canonical NCBI taxon "
        "and ID before looking up data elsewhere. If nothing matches, try a shorter or "
        "differently spelled name.",
        {"type": "object", "properties": {
            "prokaryote_name": {"type": "string",
                               "description": "Organism or strain name, e.g. 'Escherichia coli K-12'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}},
         "required": ["prokaryote_name"], "additionalProperties": False},
        get_ncbi_strain_names,
    ),
]
