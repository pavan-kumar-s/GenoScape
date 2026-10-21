"""Your own tools. run_live.py adds everything in MY_TOOLS to the registry tools.

A tool is a name, a description the model reads, a JSON schema for its
arguments, and a Python function. Raise ToolError with a message the model
can act on when something goes wrong; it goes back to the model as a result.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import os
from html import unescape
from pathlib import Path

from agentbook import Tool, ToolError  # noqa: F401  (ToolError is for your tools)
from agentbook.registry import STATUS, default_registry  # noqa: F401

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_IDCONV_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
BACDIVE_API = "https://api.bacdive.dsmz.de/v2"
CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
MEDIA_RECORDS_DIR = Path(__file__).resolve().parents[1] / "data" / "strain_media"


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


def _bacdive_get(path_or_url: str) -> dict:
    """GET one BacDive v2 endpoint (relative path, or the full `next` URL it returns
    for pagination). Raises ToolError on any failure."""
    url = path_or_url if path_or_url.startswith("http") else f"{BACDIVE_API}/{path_or_url}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise ToolError("BacDive rate-limited this request (HTTP 429). Wait a moment "
                            "and retry.") from e
        raise ToolError(f"BacDive returned HTTP {e.code} for {path_or_url!r}.") from e
    except urllib.error.URLError as e:
        raise ToolError(f"Could not reach the BacDive API ({e.reason}). Check network "
                        "access and retry.") from e


def _ncbi_taxids(record: dict) -> list[dict]:
    """The record's NCBI tax id entries, normalized to a list of
    {'NCBI tax id': ..., 'Matching level': 'species'|'strain'}."""
    entry = (record.get("General") or {}).get("NCBI tax id")
    if entry is None:
        return []
    return entry if isinstance(entry, list) else [entry]


def _collect_bacdive_ids(first_page: dict, limit: int) -> list[int]:
    """Page through a BacDive taxon search ('next' is a full URL) up to `limit` ids."""
    ids, page = [], first_page
    while True:
        ids.extend(page.get("results") or [])
        if len(ids) >= limit or not page.get("next"):
            break
        page = _bacdive_get(page["next"])
    return ids[:limit]


def get_bacdive_details(scientific_name: str, ncbi_taxid: str | None = None,
                        max_results: int = 50) -> dict:
    """Search BacDive by taxonomic name, fetch full records, and -- if `ncbi_taxid` is
    given -- keep only the record(s) whose NCBI mapping includes that id (species- or
    strain-level). Returns each match's BacDive ID, its NCBI tax id entries, and the
    complete record (General, Morphology, Culture and growth conditions, Physiology
    and metabolism, Sequence information, Genome-based predictions, etc.)."""
    parts = [p for p in scientific_name.strip().split() if p.lower() != "subsp."]
    if not parts or len(parts) > 3:
        raise ToolError(f"'{scientific_name}' should be 1-3 taxonomic units: genus, "
                        "optionally species epithet, optionally subspecies (e.g. "
                        "'Escherichia coli'). A strain designation like 'K-12' or 'W' "
                        "won't match here -- use ncbi_taxid to narrow to a strain instead.")
    path = "/".join(urllib.parse.quote(p, safe="") for p in parts)
    search = _bacdive_get(f"taxon/{path}")
    if "count" not in search:
        raise ToolError(f"BacDive search failed for '{scientific_name}': "
                        f"{search.get('message') or search.get('title') or search}")
    if search["count"] == 0:
        raise ToolError(f"No BacDive records found for '{scientific_name}'. Try just the "
                        "genus, or check the spelling.")

    ids = _collect_bacdive_ids(search, max_results)
    fetched = _bacdive_get("fetch/" + ";".join(str(i) for i in ids))
    results = fetched.get("results") or {}
    records = list(results.values()) if isinstance(results, dict) else list(results)
    strains = [{"bacdive_id": (r.get("General") or {}).get("BacDive-ID"),
               "ncbi_tax_ids": _ncbi_taxids(r), "record": r} for r in records]

    if ncbi_taxid is not None:
        matched = [s for s in strains if any(str(t.get("NCBI tax id")) == str(ncbi_taxid)
                                             for t in s["ncbi_tax_ids"])]
        if not matched:
            raise ToolError(f"None of the {len(strains)} BacDive records fetched for "
                            f"'{scientific_name}' matched NCBI tax id {ncbi_taxid} (out of "
                            f"{search['count']} total hits). Retry with a higher max_results "
                            "to search further down the list, or omit ncbi_taxid to see all "
                            "fetched records' tax ids.")
        strains = matched
    return {"query": {"scientific_name": scientific_name, "ncbi_taxid": ncbi_taxid},
            "total_found": search["count"], "returned": len(strains), "strains": strains}


def _fetch_bacdive_record(bacdive_id: int | str) -> dict:
    """Fetch one full BacDive record by its BacDive-ID, re-fetching rather than trusting
    a record passed in from an earlier step (tool results only round-trip as text)."""
    fetched = _bacdive_get(f"fetch/{bacdive_id}")
    results = fetched.get("results") or {}
    record = results.get(str(bacdive_id)) if isinstance(results, dict) else (
        results[0] if results else None)
    if not record:
        raise ToolError(f"BacDive-ID {bacdive_id} was not found.")
    return record


def _is_type_strain(record: dict) -> bool:
    tax = record.get("Name and taxonomic classification") or {}
    tax = tax[0] if isinstance(tax, list) else tax
    return str((tax or {}).get("type strain", "")).strip().lower() == "yes"


def pick_top_bacdive_strain(prokaryote_name: str, max_results: int = 50) -> dict:
    """Resolve a prokaryote name all the way to ONE BacDive strain: look it up in NCBI
    Taxonomy, search BacDive for that species, and pick a single record -- preferring one
    whose NCBI mapping matches the resolved NCBI tax id, then the type strain, then the
    lowest BacDive-ID (for a deterministic pick among equally good candidates). Feed the
    returned bacdive_id into get_bacdive_carbon_source_growth or get_bacdive_reference_text."""
    ncbi = get_ncbi_strain_names(prokaryote_name, max_results=1)
    taxon = ncbi["taxa"][0]
    ncbi_taxid, scientific_name = taxon["ncbi_taxid"], taxon["scientific_name"]
    # BacDive's taxon search takes genus [+ species epithet], not a strain suffix.
    species_name = " ".join(scientific_name.split()[:2])

    ncbi_taxid_matched = True
    try:
        bd = get_bacdive_details(species_name, ncbi_taxid=ncbi_taxid, max_results=max_results)
    except ToolError:
        ncbi_taxid_matched = False
        bd = get_bacdive_details(species_name, max_results=max_results)

    type_strains = [s for s in bd["strains"] if _is_type_strain(s["record"])]
    pool = type_strains or bd["strains"]
    top = min(pool, key=lambda s: s["bacdive_id"])

    result = {
        "prokaryote_name": prokaryote_name,
        "ncbi_taxid": ncbi_taxid,
        "scientific_name": scientific_name,
        "bacdive_id": top["bacdive_id"],
        "ncbi_tax_ids": top["ncbi_tax_ids"],
        "is_type_strain": _is_type_strain(top["record"]),
        "ncbi_taxid_matched": ncbi_taxid_matched,
        "candidates_considered": bd["returned"],
        "total_bacdive_hits": bd["total_found"],
    }
    if not type_strains and bd["returned"] < bd["total_found"]:
        result["note"] = (
            f"No type strain found among the {bd['returned']} of {bd['total_found']} candidates "
            "scanned (BacDive pages by ascending BacDive-ID, not by relevance), so this pick is "
            "just the lowest-ID candidate in that window, not necessarily the type strain or the "
            "'best' one. Re-run with a higher max_results if the type strain specifically matters.")
    return result


GROWTH_UTILIZATION_KINDS = {"carbon source", "growth"}


def get_bacdive_carbon_source_growth(bacdive_id: int) -> dict:
    """Return one BacDive strain's growth-relevant metabolite tests: entries whose `kind
    of utilization tested` is 'carbon source' or 'growth' (BacDive curators use either
    label for the same kind of test, depending on the source). Each result has the
    metabolite, its Chebi-ID, whether it supported growth (+ -> true), and the `ref` id
    of the source backing it. Pass this strain's bacdive_id to
    save_carbon_source_media_records to resolve and save all of them at once."""
    record = _fetch_bacdive_record(bacdive_id)
    phys = record.get("Physiology and metabolism") or {}
    entries = phys.get("metabolite utilization") or []
    if isinstance(entries, dict):
        entries = [entries]
    if not entries:
        raise ToolError(f"BacDive-ID {bacdive_id} has no metabolite utilization data at all.")
    tests = [
        {"metabolite": e.get("metabolite"), "chebi_id": e.get("Chebi-ID"),
         "growth": e.get("utilization activity") == "+",
         "utilization_activity": e.get("utilization activity"), "ref": e.get("@ref")}
        for e in entries if e.get("kind of utilization tested") in GROWTH_UTILIZATION_KINDS
    ]
    if not tests:
        kinds_present = sorted({e.get("kind of utilization tested") for e in entries})
        raise ToolError(f"BacDive-ID {bacdive_id} has no 'carbon source' or 'growth' "
                        f"utilization tests (only: {', '.join(kinds_present)}).")
    return {"bacdive_id": bacdive_id, "n_tests": len(tests), "tests": tests}


def _strip_html(html: str) -> str:
    """Rough HTML -> text: drop script/style blocks and tags, unescape entities, collapse
    whitespace. Not a real parser -- good enough to read prose, not tables or figures."""
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = unescape(re.sub(r"(?s)<[^>]+>", " ", html))
    return re.sub(r"[ \t]+", " ", text).strip()


def _fetch_json_quiet(url: str) -> dict | None:
    """GET JSON from a best-effort enrichment API (Crossref/Unpaywall); None on any failure
    so a down/slow third-party service degrades the result instead of failing the tool."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "genoscape-agentbook-tool"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_bytes(url: str) -> tuple[str | None, bytes | None]:
    """GET raw bytes and the Content-Type of a (possibly paywalled or blocked) URL.
    (None, None) on any failure rather than raising, since the caller still has metadata."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (genoscape-agentbook)"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            return resp.headers.get("Content-Type"), resp.read()
    except Exception:
        return None, None


def _pdf_text(data: bytes, max_pages: int = 40) -> str:
    try:
        import pypdf
    except ImportError:
        return "[PDF content: install pypdf (`pip install pypdf`) to extract its text.]"
    import io
    reader = pypdf.PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])
    return text.strip() or "[PDF had no extractable text layer (likely a scanned image).]"


def _idconv(doi: str) -> dict:
    """DOI -> {'pmcid', 'pmid'} via NCBI's ID Converter, or {} if the DOI isn't in PMC."""
    params = {"ids": doi, "format": "json", "tool": "genoscape_agentbook"}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    url = f"{PMC_IDCONV_API}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        return {}
    record = (data.get("records") or [{}])[0]
    if record.get("status") == "error" or not record.get("pmcid"):
        return {}
    return {"pmcid": record["pmcid"], "pmid": record.get("pmid")}


def get_pubmed_reference_text(doi: str) -> dict:
    """Resolve a DOI to its full text in PubMed Central, if it's in the PMC Open Access /
    author-manuscript set. Converts the DOI to a PMCID via NCBI's ID Converter, then
    fetches and flattens the JATS XML body via efetch (db=pmc). Returns `text` empty with
    `status` explaining why when the DOI has no PMCID -- true for most paywalled or
    non-PMC-deposited articles -- in which case get_bacdive_reference_text's
    Crossref/Unpaywall/direct-fetch path is the fallback."""
    ids = _idconv(doi)
    if not ids.get("pmcid"):
        return {"doi": doi, "pmcid": None, "pmid": None, "text": "",
                "status": "This DOI has no PMCID: the article is not deposited in PubMed "
                         "Central, so its full text is not available this way."}
    xml_text = _ncbi_get("efetch", db="pmc", id=ids["pmcid"], rettype="full", retmode="xml")
    root = ET.fromstring(xml_text)
    body = root.find(".//body")
    text = ""
    if body is not None:
        text = re.sub(r"\s+", " ", " ".join(t.strip() for t in body.itertext() if t.strip())).strip()
    return {
        "doi": doi, "pmcid": ids["pmcid"], "pmid": ids.get("pmid"), "text": text[:20000],
        "status": "ok" if text else "PMCID found but its XML had no <body> text (front-matter-only stub).",
    }


def get_bacdive_reference_text(bacdive_id: int, ref_id: int) -> dict:
    """Fetch the text behind one of a BacDive record's literature references (its `@ref` /
    `@id`), to read the primary source behind a BacDive-curated fact -- e.g. the exact
    media recipe a growth claim came from. For a DOI reference: pulls title/abstract from
    Crossref, and (if the UNPAYWALL_EMAIL environment variable is set to a real contact
    email) looks up an open-access copy via Unpaywall; otherwise fetches the DOI's landing
    page directly. For a plain URL reference: fetches that page directly. Only open or
    otherwise unrestricted pages can actually be read -- a paywalled paper comes back as
    title/abstract only, with `text` empty or sparse; check `status`/`open_access`."""
    record = _fetch_bacdive_record(bacdive_id)
    refs = record.get("Reference") or []
    if isinstance(refs, dict):
        refs = [refs]
    ref = next((r for r in refs if r.get("@id") == ref_id), None)
    if ref is None:
        raise ToolError(f"BacDive-ID {bacdive_id} has no reference @id={ref_id}. Available "
                        f"ref ids: {sorted(r.get('@id') for r in refs)}.")

    location = (ref.get("doi/url") or "").strip()
    result = {"bacdive_id": bacdive_id, "ref_id": ref_id, "authors": ref.get("authors"),
              "title": ref.get("title"), "doi_or_url": location or None}
    if not location:
        result.update(status="This reference has no doi/url on record; nothing to fetch.", text="")
        return result

    if location.startswith("http"):
        fetch_url = location
    else:
        doi = location
        result["doi"] = doi
        crossref = _fetch_json_quiet(f"{CROSSREF_API}/{urllib.parse.quote(doi)}")
        if crossref:
            msg = crossref.get("message", {})
            result["crossref_title"] = (msg.get("title") or [None])[0]
            if msg.get("abstract"):
                result["abstract"] = _strip_html(msg["abstract"])
        fetch_url = f"https://doi.org/{doi}"
        email = os.environ.get("UNPAYWALL_EMAIL")
        if email:
            oa = _fetch_json_quiet(f"{UNPAYWALL_API}/{doi}?email={urllib.parse.quote(email)}")
            best = (oa or {}).get("best_oa_location") or {}
            fetch_url = best.get("url_for_pdf") or best.get("url") or fetch_url
            result["open_access"] = bool(oa and oa.get("is_oa"))
        else:
            result["open_access"] = None
            result["note"] = "Set UNPAYWALL_EMAIL (a real contact email) to also check for an open-access copy."

    content_type, body = _fetch_bytes(fetch_url)
    result["fetched_from"] = fetch_url
    if not body:
        result["text"] = ""
        result["status"] = "Could not fetch this URL; only the metadata above is available."
    elif content_type and "pdf" in content_type.lower():
        result["content_type"] = "pdf"
        result["text"] = _pdf_text(body)
    else:
        result["content_type"] = content_type or "text/html"
        result["text"] = _strip_html(body.decode("utf-8", "ignore"))[:12000]
    return result


EXTRACTION_SYSTEM = (
    "You extract culture-medium / growth-condition information from one scientific "
    "paper's text. Report only what the text actually states -- never guess, infer a "
    "textbook default, or fill in a gap. Respond with ONLY a JSON object, no other text "
    "and no markdown fences, matching exactly this shape:\n"
    '{"found": bool, "media": [{"name": string|null, '
    '"ingredients": [{"name": string, "amount": string|null}], '
    '"temperature": string|null, "ph": string|null, "notes": string|null}]}\n'
    'If the text does not describe a growth medium at all, return {"found": false, "media": []}.'
)


def _extraction_model():
    """The model used to read a paper and extract media details: OpenRouter if
    OPENROUTER_API_KEY is set, otherwise Anthropic directly via ANTHROPIC_API_KEY.
    Raises ToolError if neither is set -- this step needs a real model, unlike every
    other tool in this file."""
    model_name = os.environ.get("EXTRACTION_MODEL")
    if os.environ.get("OPENROUTER_API_KEY"):
        from agentbook.models import OpenAIModel
        return OpenAIModel(model_name or "anthropic/claude-haiku-4.5",
                           base_url="https://openrouter.ai/api/v1",
                           api_key=os.environ["OPENROUTER_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        from agentbook.models import AnthropicModel
        return AnthropicModel(model_name or "claude-haiku-4.5")
    raise ToolError("extract_media_composition needs a model to read the paper text: set "
                    "OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env.")


def extract_media_composition(text: str) -> dict:
    """Upload the whole paper text to an LLM and have it extract the culture-medium /
    growth-condition details it actually states, as structured JSON: for each medium
    described, its name, ingredients (name + amount), temperature, pH, and any other
    notes. Call this ONCE per paper, not once per carbon source -- the same paper is
    often the reference behind several carbon-source tests on the same strain, and its
    medium doesn't change between them. Returns {"found": false, "media": []} if the
    text doesn't describe a medium at all; raises ToolError if there's no model
    configured, or if the model's reply isn't valid JSON."""
    if not text.strip():
        raise ToolError("No text to extract from.")
    model = _extraction_model()
    messages = [{"role": "user", "content": f"Paper text:\n\n{text}"}]
    response = model.complete(EXTRACTION_SYSTEM, messages, tools=[], max_tokens=1500)
    raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", response.text.strip())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ToolError(f"Extraction model did not return valid JSON: {raw[:300]}") from e
    return parsed


def _best_genome_accession(record: dict, ncbi_taxid: str | None = None) -> str | None:
    """The record's best INSDC genome accession: prefer one whose own NCBI tax ID matches
    `ncbi_taxid`, then the highest BacDive match `score`."""
    genomes = (record.get("Sequence information") or {}).get("Genome sequences") or []
    if isinstance(genomes, dict):
        genomes = [genomes]
    if not genomes:
        return None
    if ncbi_taxid is not None:
        matching = [g for g in genomes if str(g.get("NCBI tax ID")) == str(ncbi_taxid)]
        genomes = matching or genomes
    best = max(genomes, key=lambda g: g.get("score") or 0)
    return best.get("INSDC accession")


def _slugify(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", text)).strip("_").lower() or "unknown"


def _resolve_reference(bacdive_id: int, record: dict, ref_id: int) -> dict:
    """Track down the text and media-composition candidates behind one reference: PMC
    full text first, then Crossref/Unpaywall/direct fetch, then scan whatever text came
    back for media lines. Call once per ref_id -- several carbon-source tests can share
    the same reference, and the medium it describes doesn't depend on which metabolite
    is being asked about."""
    refs = record.get("Reference") or []
    if isinstance(refs, dict):
        refs = [refs]
    ref = next((r for r in refs if r.get("@id") == ref_id), None)
    if ref is None:
        raise ToolError(f"BacDive-ID {bacdive_id} has no reference @id={ref_id}.")
    location = (ref.get("doi/url") or "").strip()

    text, provenance = "", {"ref_id": ref_id, "reference_title": ref.get("title"),
                            "doi_or_url": location or None, "text_source": "none"}
    if location and not location.startswith("http"):
        provenance["doi"] = location
        pubmed = get_pubmed_reference_text(location)
        if pubmed.get("text"):
            text, provenance["text_source"] = pubmed["text"], "pubmed_central"
            provenance["pmcid"] = pubmed.get("pmcid")
        else:
            fallback = get_bacdive_reference_text(bacdive_id, ref_id)
            if fallback.get("text"):
                text, provenance["text_source"] = fallback["text"], fallback.get("content_type", "web")
            provenance["pubmed_status"] = pubmed.get("status")
    elif location:
        fallback = get_bacdive_reference_text(bacdive_id, ref_id)
        if fallback.get("text"):
            text, provenance["text_source"] = fallback["text"], fallback.get("content_type", "web")

    full_media = []
    if text:
        try:
            full_media = extract_media_composition(text).get("media") or []
        except ToolError as e:
            provenance["media_extraction_status"] = f"extraction failed: {e}"
    if not full_media and "media_extraction_status" not in provenance:
        provenance["media_extraction_status"] = (
            "no text available to scan" if not text else
            "model found no medium described in the text")

    return {"full_media": full_media, "provenance": provenance}


def save_carbon_source_media_records(bacdive_id: int, strain_name: str,
                                     ncbi_id: str | None = None) -> dict:
    """Resolve EVERY carbon-source/growth test for one BacDive strain
    (get_bacdive_carbon_source_growth) and save them all into a single JSON file:
    data/strain_media/<strain_name>_<ncbi_id>.json.

    For each distinct reference behind those tests (several tests often share one):
    track down its text (PubMed Central full text first, falling back to
    Crossref/Unpaywall/direct fetch if the reference isn't in PMC), then have an LLM
    extract its media composition (extract_media_composition) -- resolved once per
    reference, not once per carbon source, so tests sharing a reference get identical,
    consistent results instead of independent (and possibly flaky) re-fetches and
    re-extractions. The file has strain_name, ncbi_id, strain_id, genome at the top
    level, and one entry per carbon source under `carbon_sources` (carbon_source,
    growth, full_media, provenance). `full_media` is only ever what a paper's own text
    yielded -- left empty (never silently filled from BacDive's own 'culture medium'
    field) when nothing was found, so every entry's source stays honest."""
    record = _fetch_bacdive_record(bacdive_id)
    tests = get_bacdive_carbon_source_growth(bacdive_id)["tests"]

    resolved_by_ref: dict[int, dict] = {}
    carbon_sources = []
    for test in tests:
        ref_id = test["ref"]
        if ref_id not in resolved_by_ref:
            resolved_by_ref[ref_id] = _resolve_reference(bacdive_id, record, ref_id)
        resolved = resolved_by_ref[ref_id]
        carbon_sources.append({"carbon_source": test["metabolite"], "growth": test["growth"],
                              "full_media": resolved["full_media"], "provenance": resolved["provenance"]})

    out = {
        "strain_name": strain_name,
        "ncbi_id": ncbi_id,
        "strain_id": bacdive_id,
        "genome": _best_genome_accession(record, ncbi_id),
        "carbon_sources": carbon_sources,
    }

    MEDIA_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = MEDIA_RECORDS_DIR / f"{_slugify(strain_name)}_{_slugify(str(ncbi_id))}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "record": out}


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
    Tool(
        "get_bacdive_details",
        "Resolve a prokaryote to its BacDive record(s). Searches BacDive by taxonomic name "
        "(genus, optionally species epithet, optionally subspecies -- NOT a strain "
        "designation) and returns full records: BacDive ID, NCBI tax id mapping, and all "
        "sections (morphology, culture/growth conditions, physiology, isolation source, "
        "sequence info, genome-based predictions, literature). Pass `ncbi_taxid` (e.g. from "
        "get_ncbi_strain_names) to keep only the record(s) matching that exact NCBI taxon -- "
        "species- or strain-level. If nothing matches, retry with a higher max_results: "
        "BacDive pages results by ascending BacDive-ID, so a specific strain may not be in "
        "the default first 50.",
        {"type": "object", "properties": {
            "scientific_name": {"type": "string",
                               "description": "e.g. 'Escherichia coli' (genus [species [subspecies]])."},
            "ncbi_taxid": {"type": "string",
                          "description": "NCBI taxonomy ID to narrow to, e.g. '566546'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}},
         "required": ["scientific_name"], "additionalProperties": False},
        get_bacdive_details,
    ),
    Tool(
        "pick_top_bacdive_strain",
        "Resolve a prokaryote name end-to-end to ONE BacDive strain (NCBI Taxonomy lookup "
        "-> BacDive search -> pick the NCBI-matched / type / lowest-ID record), so you don't "
        "have to pick among get_bacdive_details's candidates yourself. Returns bacdive_id "
        "plus the NCBI taxid and scientific name used to get there. Use the returned "
        "bacdive_id with get_bacdive_carbon_source_growth and get_bacdive_reference_text.",
        {"type": "object", "properties": {
            "prokaryote_name": {"type": "string",
                               "description": "Organism or strain name, e.g. 'Escherichia coli K-12'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50}},
         "required": ["prokaryote_name"], "additionalProperties": False},
        pick_top_bacdive_strain,
    ),
    Tool(
        "get_bacdive_carbon_source_growth",
        "Return one BacDive strain's growth-relevant metabolite tests: entries whose "
        "'kind of utilization tested' is 'carbon source' or 'growth'"
        " -- BacDive curators use either "
        "label for the same kind of test depending on the source. Each result has the "
        "tested metabolite, whether it supported growth (true/false from the +/- "
        "utilization activity), and the reference id the test came from. Pass this "
        "strain's bacdive_id into save_carbon_source_media_records to resolve and save "
        "all of them at once. Fails if the strain has neither kind of utilization data -- "
        "the error lists which other kinds it does have, if any.",
        {"type": "object", "properties": {
            "bacdive_id": {"type": "integer", "description": "From pick_top_bacdive_strain "
                           "or get_bacdive_details."}},
         "required": ["bacdive_id"], "additionalProperties": False},
        get_bacdive_carbon_source_growth,
    ),
    Tool(
        "get_bacdive_reference_text",
        "Fetch the text behind one literature reference (`@ref`/`@id`) of a BacDive "
        "record -- the primary source behind a BacDive-curated fact, e.g. the paper a "
        "growth or media claim came from. Resolves a DOI via Crossref (title/abstract) and, "
        "if UNPAYWALL_EMAIL is set, via Unpaywall for an open-access copy; a plain URL "
        "reference is fetched directly. Only open-access or otherwise unrestricted pages "
        "can actually be read -- a paywalled paper comes back with title/abstract but an "
        "empty or sparse `text`, flagged in `status`/`open_access`.",
        {"type": "object", "properties": {
            "bacdive_id": {"type": "integer"},
            "ref_id": {"type": "integer", "description": "A reference '@ref' id, e.g. from "
                      "get_bacdive_carbon_source_growth's 'ref' field."}},
         "required": ["bacdive_id", "ref_id"], "additionalProperties": False},
        get_bacdive_reference_text,
    ),
    Tool(
        "get_pubmed_reference_text",
        "Resolve a DOI to its full text in PubMed Central (PMC), if the article is in "
        "PMC's Open Access / author-manuscript set. Converts the DOI to a PMCID via NCBI's "
        "ID Converter and fetches the structured JATS full text via efetch -- more "
        "reliable than scraping a publisher page when it works, but only works for "
        "articles actually deposited in PMC. Returns `text` empty with `status` explaining "
        "why otherwise (most paywalled or non-PMC-deposited articles); fall back to "
        "get_bacdive_reference_text's Crossref/Unpaywall path in that case.",
        {"type": "object", "properties": {
            "doi": {"type": "string", "description": "e.g. '10.1099/ijsem.0.004332' "
                   "(from a BacDive reference's doi/url field)."}},
         "required": ["doi"], "additionalProperties": False},
        get_pubmed_reference_text,
    ),
    Tool(
        "extract_media_composition",
        "Upload the whole paper text (typically the `text` field from "
        "get_bacdive_reference_text or get_pubmed_reference_text) to an LLM and have it "
        "extract the culture-medium / growth-condition details it actually states, as "
        "structured JSON: for each medium described, its name, ingredients (name + "
        "amount), temperature, pH, and notes. Call this ONCE per paper, not once per "
        "carbon source -- the same paper is often the reference for several "
        "carbon-source tests on the same strain. Needs OPENROUTER_API_KEY or "
        "ANTHROPIC_API_KEY set; returns {'found': false, 'media': []} if the text "
        "doesn't describe a medium at all.",
        {"type": "object", "properties": {
            "text": {"type": "string"}},
         "required": ["text"], "additionalProperties": False},
        extract_media_composition,
    ),
    Tool(
        "save_carbon_source_media_records",
        "Resolve EVERY carbon-source/growth test for one BacDive strain and save them "
        "all into a SINGLE JSON file: data/strain_media/<strain_name>_<ncbi_id>.json. "
        "For each distinct reference (several carbon sources often share one): tries "
        "PubMed Central full text first, falls back to Crossref/Unpaywall/direct fetch "
        "if the reference isn't in PMC, then has an LLM extract its media composition "
        "once and reuses that for every carbon source citing it. Returns the saved file "
        "path and the record: strain_name, ncbi_id, strain_id, genome (best INSDC "
        "accession) at the top level, and a `carbon_sources` list with one entry per "
        "carbon source (carbon_source, growth, full_media -- structured media found IN "
        "THE PAPER, empty rather than guessed if none were found -- and a provenance "
        "block). Needs OPENROUTER_API_KEY or ANTHROPIC_API_KEY set.",
        {"type": "object", "properties": {
            "bacdive_id": {"type": "integer", "description": "From pick_top_bacdive_strain "
                           "or get_bacdive_details."},
            "strain_name": {"type": "string", "description": "The organism/strain name used "
                           "to resolve it, e.g. from pick_top_bacdive_strain's "
                           "'scientific_name'. Used to name the output file."},
            "ncbi_id": {"type": "string", "description": "NCBI taxonomy ID for this strain, "
                       "e.g. from pick_top_bacdive_strain. Used to name the output file."}},
         "required": ["bacdive_id", "strain_name"], "additionalProperties": False},
        save_carbon_source_media_records,
    ),
]
