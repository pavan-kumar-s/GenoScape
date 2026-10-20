```markdown
# GenoScape, Team GenoScape


Members: Pavan Kumar
Repo: https://github.com/pavan-kumar-s/GenoScape


## 1. Task
- Building a ML/DL ready database required for Genotype + Environment to Phenotype (growth) prediction
- An ML/DL model trained on such database could help researchers to cultivate microbes present in environment
- **Example:** 
  - **Input:** Name or NCBI Taxonomy ID of a prokaryote (e.g., *Escherichia coli* K-12 or *Pseudomonas putida*).
  - **Output:** A structured JSON/CSV table of experimentally validated genetic perturbations and environmental perturbations paired with growth phenotypes.


## 2. Why an agent?
- **Dynamic query formulation:** Perturbation data is described inconsistently across sources (e.g., "Biolog PM1", "auxotrophy"). An agent must iteratively expand synonyms, resolve strain/gene identifiers, and refine search queries based on intermediate findings.
- **Multi-source routing & cross-referencing:** No single database holds complete genotype-environment-phenotype triads. The agent must decide when to query structured databases versus when to mine unstructured full-text literature (PubMed Central) to fill missing media composition or mutation details.
- **Schema normalization:** Experimental results in papers and databases use varying units and formats; the agent must extract, validate, and standardize them into a unified ML-ready schema.


## 3. Tools and data (3-6 tools)
- **`resolve_taxon_and_genome`**: Resolves organism names to canonical NCBI TaxIDs, strain identifiers, and reference genome accessions.
  - **In:** Organism name (string).
  - **Out:** TaxID, canonical strain name, assembly accession (JSON).
  - **Data source:** NCBI Entrez / Datasets REST API.
  - **Access checked:** Yes.
- **`query_mediadive`**: Retrieves known cultivation media recipes.
  - **In:** TaxID or species name (string).
  - **Out:** Structured list of media recipes.
  - **Data source:** DSMZ BacDive & MediaDive REST APIs.
  - **Access checked:** Yes.
- **`query_fitness_browser`**: Fetches genome-wide transposon mutagenesis (RB-TnSeq) fitness scores across dozens of carbon/nitrogen/stress conditions.
  - **In:** Organism name or TaxID (string).
  - **Out:** Gene locus tags, condition descriptions, and log2 fitness ratios (JSON/TSV).
  - **Data source:** LBNL Fitness Browser (fit.genomics.lbl.gov) API / static SQLite dump.
  - **Access checked:** Yes.
- **`search_pubmed_literature`**: Searches scientific literature for experimental growth and gene perturbation studies when structured databases lack coverage.
  - **In:** Search query with organism and perturbation keywords (string).
  - **Out:** List of PMIDs/PMCIDs, titles, and abstracts/full-text XML snippets (JSON).
  - **Data source:** NCBI E-utilities / Europe PMC REST API.
  - **Access checked:** Yes.
- **`extract_and_validate_records`**: Parses unstructured text/tables into the strict GenoScape Pydantic schema and validates that genotype, environment, and growth phenotype fields are present.
  - **In:** Raw text snippet or database payload (string) + source ID.
  - **Out:** Validated list of `PerturbationRecord` objects (JSON).
  - **Data source:** Local Python tool / Pydantic validator.
  - **Access checked:** Yes.


## 4. Baseline (Milestone III, Tue 5 PM)
- Prompt a single LLM with the prokaryote name and ask it to output a JSON list of known genetic and environmental perturbations and resulting growth phenotypes purely from parametric memory, and database lookups without literature search.


## 5. Evaluation (Milestone IV, Wed 11 AM)
- **Test cases:** 10 diverse prokaryotes with known ground-truth records (5 well-studied model organisms like *E. coli*, *B. subtilis*, *P. aeruginosa*, *S. elongatus*, *C. crescentus*; 5 non-model/environmental prokaryotes tracked in BacDive/Fitness Browser like *P. putida*, *D. vulgaris*, *Z. mobilis*, *M. extorquens*, *B. thetaiotaomicron*).
- **Metric:** 
  1. **Schema Validity & Grounding Rate (%):** Percentage of extracted records that pass Pydantic validation and contain a verifiable source ID (PMID, BacDive ID, or Fitness Browser exp ID).
  2. **Recall / Coverage against Gold Set:** F1/Recall of known genetic + media perturbations on a curated gold subset of 50 ground-truth perturbation-phenotype pairs.
- **Comparison:** Baseline vs. Single-shot PubMed script vs. Full GenoScape Multi-Tool Agent.


## 6. Demo (Milestone V, Wed 1 PM)
- **Live input:** Run the agent live on an example prokaryote.
- **What the audience will see:** 
  1. The agent resolving the strain ID via NCBI.
  2. The agent querying BacDive/Fitness Browser for baseline media and mutant fitness profiles, then autonomously formulating PubMed queries to find specific knockout growth assays.
  3. The final compiled, ML-ready DataFrame/JSON table displaying paired `[Gene Perturbation | Media/Environment | Growth Phenotype | Source PMID/DB ID]` alongside a side-by-side comparison showing the baseline.


## 7. Risks, limits, roles
- **Biggest risk:** Full-text papers often bury Biolog or growth curve data in supplementary PDFs/Excel files, and API rate limits or noisy abstract text may lead to incomplete environment specifications.
- **Fallback if you are behind on Tuesday at noon:** Restrict scope to two reliable sources—**Europe PMC API** (abstracts + open-access full text) and **Fitness Browser / BacDive**—.
- **Limits:** Max 10 agent steps per organism; max 30,000 input tokens per run; expected API spend < $50 total across evaluation runs.


```
