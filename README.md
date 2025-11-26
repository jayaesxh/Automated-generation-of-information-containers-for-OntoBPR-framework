# Automated Generation of ICDD Containers for Building Permit Applications

This repository contains the code for an end-to-end pipeline that:

1. Ingests a building-permit submission (ZIP or folder) for a single case.
2. Builds an ISO 21597-compliant ICDD container (index.rdf, payload documents, linkset, ontology resources).
3. Preprocesses all documents into text chunks (JSONL + manifest) for retrieval.
4. Uses a Large Language Model (LLM) to extract key application and building fields.
5. Converts the extracted fields into an OntoBPR knowledge graph fragment (OntoBPR.ttl).
6. Packs everything into a single `.icdd` container and validates the structure with SHACL.

The result is an ICDD container that not only bundles the original permit documents,
but also contains an RDF graph conformant to OntoBPR, ready for use in downstream
reasoning or graph queries.

## 1. Repository structure

```text
icdd-rag-pipeline2/
├─ driver_upload.py          # main entry: runs whole pipeline for all submissions in ./upload
├─ ontobpr_llm.py            # LLM-based schema extraction + mapping to OntoBPR
├─ field_schema.py           # field definitions (application + building)
├─ llm_backend.py            # wraps the Hugging Face LLM client
├─ rag_lightrag.py           # (optional) LightRAG integration for graph-based retrieval
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ notebooks/
│   └─ Automated_Generation_of_ICDD.ipynb   # optional, for demonstration / experimentation
├─ static_resources/
│   └─ ontology_resources/
│       ├─ Container.rdf
│       ├─ Linkset.rdf
│       ├─ ExtendedLinkset.rdf
│       ├─ Container.shapes.ttl
│       ├─ Part1ClassesCheck.shapes.rdf
│       ├─ Part2ClassesCheck.shapes.ttl
│       └─ ...
├─ upload/
│   └─ .gitkeep          # place your case ZIPs / folders here
└─ output/
    └─ ...               # generated cases + ICDD containers
```

## 2. Installation

1. Clone the repository:

   ```bash
   git clone <YOUR_REPO_URL>.git
   cd icdd-rag-pipeline2
   ```

2. Create a virtual environment (optional but recommended) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure you have a working Python 3.10+ environment with CUDA (if you want to run LLMs locally).

## 3. LLM configuration (Hugging Face Inference API)

The pipeline uses a chat-based LLM via Hugging Face Inference API
(e.g. `Qwen/Qwen2.5-7B-Instruct`) to extract structured fields from the documents.

1. Create a Hugging Face account and generate an API token.
2. Set the token as environment variable before running the pipeline:

   ```bash
   export HF_TOKEN="hf_xxx_your_token_here"
   ```

3. The file `llm_backend.py` reads the `HF_TOKEN` from the environment and uses it
   to perform a chat-completion call. The LLM is instructed to output **strict JSON**
   matching the schema defined in `field_schema.py`.

## 4. Running the pipeline

1. Prepare a single-case submission as a ZIP or folder (e.g. containing all PDFs,
   forms, and letters for case `Apl_25-04543-FULL`).

2. Place it into the `upload/` directory:

   ```bash
   cp Apl_25-04543-FULL.zip upload/
   ```

3. Run the main pipeline:

   ```bash
   python driver_upload.py
   ```

   For each item in `upload/`, the script will:

   - Create `output/<CASE_ID>/` with:
     - `Payload documents/<CASE_ID>/...`
     - `Payload triples/Doc_Application_Links.rdf`
     - `Payload triples/OntoBPR.ttl` (LLM-based OntoBPR graph)
     - `Ontology resources/...` (ICDD ontologies + SHACL shapes)
     - `index.rdf` (ICDD container index)
     - `rag/chunks.jsonl` & `rag/manifest.json` (preprocessed text chunks)

   - Build and write a zipped container:
     - `output/ICDD_<CASE_ID>.icdd`

   - Run structural and coherence checks on the container.
   - Run SHACL validation using the provided shapes.

## 5. Pipeline overview

### 5.1 Upload and staging

- `driver_upload.py` scans `./upload/` for non-hidden items.
- Each item becomes a `CASE_ID` (based on filename).
- Files are copied or unzipped into `output/<CASE_ID>/Payload documents/<CASE_ID>/`.

### 5.2 ICDD container construction

- `build_index_graph()` creates `index.rdf` with:
  - A `ct:ContainerDescription` individual and `ct:Linkset` individual.
  - All documents as `ct:Document` / `ct:InternalDocument`.
  - `ct:publisher`, `ct:containsDocument`, `ct:belongsToContainer`, and `ct:containsLinkset`.

- `build_payload_linkset()` creates `Payload triples/Doc_Application_Links.rdf`
  with at least one example link between documents.

### 5.3 Preprocessing for retrieval

- `preprocess_for_rag()` (in `driver_upload.py`) reads each payload document, extracts text
  (using `pdfminer.six` for PDFs and simple previews for other formats) and writes:
  - `rag/chunks.jsonl` – one JSON record per text chunk.
  - `rag/manifest.json` – metadata per document (role, filetype, number of chunks, etc.).

### 5.4 Experimental LightRAG integration

- `rag_lightrag.py` contains code to:
  - Build a LightRAG index over `rag/chunks.jsonl`.
  - Optionally retrieve focused context for the LLM.
- In environments where LightRAG or its dependencies are not fully functional,
  the pipeline falls back to using the raw chunks directly. This behavior is
  handled gracefully in `ontobpr_llm.py`.

### 5.5 LLM-based OntoBPR extraction

- `field_schema.py` defines which fields to extract (application id, authority, site address, etc.).
- `ontobpr_llm.extract_schema_with_llm()`:
  - Loads text chunks for the case.
  - Optionally calls LightRAG for additional context.
  - Builds a strict JSON template and prompts the LLM to fill it.
  - Parses the JSON (with a robust fallback if extra text is present).

- `ontobpr_llm.schema_to_ontobpr()`:
  - Maps the extracted JSON fields to OntoBPR properties.
  - Creates individuals like `ontobpr:BuildingApplication` and `ontobpr:Building`.
  - Writes the triples to `Payload triples/OntoBPR.ttl`.

### 5.6 Validation (ICDD & SHACL)

- `driver_upload.py` runs:
  - A **structural check**: ensures required files are present.
  - A **coherence check**: ensures container–linkset–document relations exist.
  - A **SHACL validation** using the shapes in `static_resources/ontology_resources/`.
    In current experiments, the only failing constraint relates to how `ct:publisher`
    is modeled; this is discussed as a modeling decision.

## 6. Limitations and further work

- LightRAG integration is experimental and may depend on specific versions of
  libraries and GPU settings.
- OCR for image-only PDFs / scanned forms is not included; such cases will have
  limited or no text extraction.
- The OntoBPR mapping currently covers a focused subset of fields (identifier,
  authority, address, use type, etc.) but can be extended to match the full
  OntoBPR/OBPA competence questions.

## 7. How to extend

- Add more `FieldSpec` entries in `field_schema.py`.
- Extend `schema_to_ontobpr()` to map new fields to OntoBPR properties.
- Add Neo4j or other graph database loaders to import `OntoBPR.ttl` for querying.
- Improve the LightRAG integration once environment constraints are resolved.
