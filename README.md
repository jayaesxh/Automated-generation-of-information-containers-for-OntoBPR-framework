# Automated Generation of ICDD Containers for Building-Permit Applications

This project provides a full-stack pipeline for transforming raw building-permit submissions into fully compliant **ISO 21597-1 information containers (ICDD)** enriched with **OntoBPR semantics**. It integrates preprocessing, retrieval-augmented generation (RAG), large-language-model (LLM) extraction, and graph mapping in a single automated workflow.



Building-permit applications typically bundle many disparate documents forms, plans, statements, scans which hinder machine reasoning. The emerging **OntoBPR ontology** and the **ISO 21597 Information Containers (ICDD)** standard aim to encode these documents and their semantics uniformly. This pipeline bridges the gap between raw submissions and a high-fidelity semantic container by:

1.  **Reading** a submission ZIP/folder and staging its contents.
2.  **Building** a Part 1 compliant ICDD container with the correct folder structure and RDF index, including `index.rdf`, `Ontology resources`, `Payload documents`, and `Payload triples` as mandated by ISO 21597.
3.  **Preprocessing** all files (PDFs, IFCs, text, etc.) into structured text chunks (`chunks.jsonl` and `manifest.json`) for retrieval.
4.  **Using a Hybrid RAG Engine** combining dense vector search (Semantic Search) and graph-based search (LightRAG) to retrieve relevant context for each query.
5.  **Employing an LLM** to extract structured information conforming to a configurable schema (`field_schema.py`) and mapping the result into OntoBPR entities and properties.
6.  **Packaging** everything original documents, extracted graphs, and linksets into a single `.icdd` file and validating it via SHACL and structural checks.

The result is a container that is both machine-readable and compliant with industry standards, ready for downstream reasoning or submission to digital permit portals.

## Repository contents

```text
icdd-rag-pipeline2/
├─ driver_upload.py          # Main entry point; orchestrates the pipeline
├─ rag_engine.py             # Hybrid RAG engine (Semantic search + LightRAG)
├─ ontobpr_llm.py            # LLM-based extraction and mapping to OntoBPR
├─ field_schema.py           # Defines fields to extract (application + building)
├─ llm_backend.py            # Wraps Hugging Face API for chat completion
├─ static_resources/
│   └─ ontology_resources/   # ISO 21597 ontologies & SHACL shapes
├─ upload/                   # Place ZIP submissions here
└─ output/                   # Pipeline output (cases + final .icdd)
```

### Key modules

| Module | Purpose |
| :--- | :--- |
| **`driver_upload.py`** | Entry script that stages each upload, builds the container, runs RAG + extraction, writes output and validation results. |
| **`rag_engine.py`** | Hybrid retrieval engine that first performs semantic search via Sentence Transformers (dense vectors) and then optionally LightRAG for graph-aware retrieval. It exposes a unified `retrieve_context` function which returns the combined context used by the LLM. |
| **`ontobpr_llm.py`** | Contains `extract_schema_with_llm` that builds a prompt using the retrieved context, calls the LLM, parses the JSON response, and maps it to an `rdflib` graph conformant to OntoBPR. |
| **`field_schema.py`** | Defines which fields to extract from the permit (e.g., `applicationIdentifier`, `applicationType`, `siteAddress`) and their types. This file can be extended to cover more OntoBPR/OBPA requirements. |
| **`llm_backend.py`** | Handles Hugging Face API calls, enabling off-the-shelf models (default `Qwen/Qwen2.5-7B-Instruct`) with deterministic decoding. |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_URL>.git
    cd icdd-rag-pipeline2
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # use .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**
    Create a `.env` file or export variables:
    ```bash
    export HF_TOKEN="hf_xxx_your_huggingface_token"
    # Optional: specify a custom Neo4j target
    export NEO4J_URI="bolt+s://<your-neo4j-host>"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="<password>"
    export NEO4J_DB="neo4j"  # or your Aura DB name
    ```

## Running the pipeline

1.  **Prepare your input:**
    Place a single ZIP or directory (containing PDFs, IFCs, etc.) in the `upload/` folder. The stem of the ZIP/directory will be used as the case ID.

2.  **Run the pipeline:**
    ```bash
    python driver_upload.py
    ```

    This will process all items in the `upload/` folder sequentially. Each run will:
    *   **Stage** the submission and copy documents into `Payload documents/<CASE_ID>/`.
    *   **Generate** `index.rdf` with a `ct:ContainerDescription` and `owl:Ontology` declaration, referencing the internal documents and linksets.
    *   **Preprocess** documents into `rag/chunks.jsonl` and `rag/manifest.json`, including PDF text extraction and fallback preview for other formats.
    *   **Build** a hybrid retrieval context using Semantic Search (dense vectors from Sentence Transformers) and LightRAG (graph-aware retrieval) to answer targeted extraction questions.
    *   **Call** the LLM to fill a strict JSON schema with extracted values and map these into an OntoBPR graph.
    *   **Create** `Doc_Application_Links.rdf` with bidirectional links between the first two documents, and write `OntoBPR.ttl` as a payload triple set.
    *   **Assemble** the final container `ICDD_<CASE_ID>.icdd`, including all original documents, the linkset, the OntoBPR graph, and all ontologies.

3.  **Review output:**
    After completion, you will find:
    *   `output/ICDD_<CASE_ID>.icdd` – a single ZIP containing `index.rdf`, `Ontology resources/`, `Payload documents/`, and `Payload triples/`.
    *   `output/<CASE_ID>/` – the working directory used during processing, containing `rag/`, intermediate files, and the unzipped container.

## Pipeline internals

### Preprocessing
All files from the submission are copied to `Payload documents/<CASE_ID>/`. The pipeline then reads each file:
*   **PDFs** are parsed using `pdfminer.six` per page. If no text is extracted, it attempts OCR via `pdf2image`/`pytesseract`; otherwise, a short preview is used.
*   **Text-based formats** (IFC, TXT, CSV, JSON, XML, YAML, XLSX, DOCX) are read using a fast preview function.
*   Each extracted chunk is recorded with metadata (`doc_iri`, `filename`, `page number`, `chunk ID`) into `chunks.jsonl`. A summary of the documents and chunk count is stored in `manifest.json`.

### Retrieval (Hybrid RAG)
The hybrid retrieval engine in `rag_engine.py` works in two layers:
1.  **Semantic Search (primary):** A dense vector index built with `SentenceTransformers` (default `all-MiniLM-L6-v2`). It maps queries and document chunks into the same embedding space and retrieves the top-K most relevant chunks.
2.  **Experimental Strategy (Graph RAG):** The architecture includes a fully integrated module for Graph-based Retrieval using `LightRAG`. While currently configured as a secondary fallback due to local compute constraints, this demonstrates the system's capability to support advanced graph reasoning.

Both retrieval outputs are combined into a single context string fed to the extraction model.

### Semantic extraction and OntoBPR mapping
The extraction module prompts a Large Language Model (LLM) with:
1.  A strict JSON schema defined in `field_schema.py`, describing application and building fields (e.g., `applicationIdentifier`, `applicationType`, `isRetrospective`, `siteAddress`, `buildingUseType`).
2.  The retrieval context returned by the hybrid RAG engine.

The model is instructed to return only valid JSON with `null` where information is unknown. The returned JSON is parsed and mapped to RDF triples using `rdflib`, creating two core resources: a `ontobpr:BuildingApplication` and a `ontobpr:Building`, together with their properties. These are written to `Payload triples/OntoBPR.ttl` and linked back to the container via `ontobpr:hasBuildingApplicationContainer`.

### Container writing and validation
Finally, the pipeline writes `index.rdf` (with `owl:Ontology` header and `ct:ContainerDescription`), `Doc_Application_Links.rdf` (the linkset), `OntoBPR.ttl` (the knowledge graph), and all ontologies into the case directory. It then zips these into `ICDD_<CASE_ID>.icdd`, ensuring that only ISO-required folders are included (no `rag/` or temporary files). Structural checks and SHACL validation are performed to ensure compliance; results are printed after each run.

## Validation and conformity
Our pipeline runs three types of validation:
1.  **Structural check** – verifies that required files exist and that the container references documents properly.
2.  **Coherence check** – ensures the index and triples are parseable as RDF.
3.  **SHACL validation** – validates the container against the official SHACL shapes for ICDD Part 1 and Part 2. Known minor violations (e.g., publisher shape) are documented and patched in our validation function.

The final output is an ISO-conformant container that can be consumed by the RUB ICDD platform or any tool supporting ISO 21597 containers.

## Future Work

This research lays the foundation for a fully automated building permit analysis system. Future development will focus on:

### 1. Automated Compliance Checking
Leveraging the generated **OntoBPR** graph to perform automated code compliance verification. By defining building regulations as **SHACL shapes** or **SPARQL queries**, the system could automatically validate the extracted building data (e.g., "Proposed Use" vs. "Zoning Constraints") against statutory requirements.

### 2. Multi-Modal RAG for Engineering Drawings
Replacing the current OCR-based text extraction with **Vision-Language Models (VLMs)** (e.g., GPT-4o, Gemini Pro Vision). This would allow the pipeline to semantically interpret complex visual data such as floor plans, elevation drawings, and site maps, extracting spatial relationships and dimensions that are currently lost in text-only processing.

### 3. Enterprise Graph Database Integration
Formalizing the storage of the generated knowledge graphs by integrating a dedicated Graph Database (e.g., **Neo4j** or **GraphDB**). This would enable:
*   Cross-case reasoning (finding patterns across multiple applications).
*   Complex graph algorithms (e.g., similarity detection between proposed and approved buildings).
*   Scalable storage for city-scale digital twins.

### 4. Ontology Expansion
Extending the current **OntoBPR** mapping to cover the full breadth of the **Building Permit Regulations (BPR)** ontology. This includes detailed modeling of fire safety standards, accessibility requirements (ISO 21542), and energy efficiency metrics.

### 5. Human-in-the-Loop Validation
Developing a semantic user interface that allows domain experts (planning officers) to visualize the extracted knowledge graph, verify the links to source documents, and correct any extraction errors before final approval.
