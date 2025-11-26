# driver_upload.py
from __future__ import annotations

import zipfile, shutil, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from ontobpr_llm import extract_schema_with_llm, schema_to_ontobpr

# --- LightRAG import made robust ---------------------------------------------
try:
    from rag_lightrag import build_lightrag_index  # real implementation
    _HAS_LIGHTRAG = True
except Exception as e:
    print(
        "[driver_upload] WARNING: rag_lightrag not available, "
        "LightRAG will be skipped:", e
    )
    _HAS_LIGHTRAG = False

    # Fallback no-op, so pipeline (LLM + OntoBPR + ICDD) still runs
    def build_lightrag_index(case_id: str, rag_dir: Path) -> None:
        return None

# ==== Paths / Namespaces ======================================================
ROOT      = Path(__file__).resolve().parent
UPLOAD    = ROOT / "upload"
OUTPUT    = ROOT / "output"
STATIC_OR = ROOT / "static_resources" / "ontology_resources"

CT  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")
LS  = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Linkset#")
ELS = Namespace("https://standards.iso.org/iso/21597/-2/ed-1/en/ExtendedLinkset#")

# ==== Data structures =========================================================
@dataclass
class DocSpec:
    iri: URIRef
    rel_filename: str  # relative to 'Payload documents/<CASE_ID>/'
    filetype: str      # 'pdf', 'xlsx', ...
    format: str        # MIME type
    name: str          # ct:name-friendly slug

@dataclass
class TextChunk:
    case_id: str
    doc_iri: str
    rel_filename: str
    name: str
    role: Optional[str]
    filetype: str
    page: int
    chunk_id: int
    text: str

# ==== Helpers =================================================================
def make_case_id(p: Path) -> str:
    return p.stem.replace(" ", "_")

def ensure_run_dir(case_id: str) -> Path:
    run_dir = OUTPUT / case_id
    # Clean old run_dir for same case to avoid confusion
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "Payload documents" / case_id).mkdir(parents=True, exist_ok=True)
    (run_dir / "Payload triples").mkdir(parents=True, exist_ok=True)
    (run_dir / "Ontology resources").mkdir(parents=True, exist_ok=True)
    (run_dir / "rag").mkdir(parents=True, exist_ok=True)

    # Copy authoritative ontology + shapes into the container
    if STATIC_OR.exists():
        for src in STATIC_OR.rglob("*"):
            if src.is_file():
                # Skip any notebook checkpoint junk at source level too
                if any(part.startswith(".ipynb_checkpoints") for part in src.parts):
                    continue
                rel = src.relative_to(STATIC_OR)
                dst = run_dir / "Ontology resources" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    else:
        print("WARNING: static ontology resources not found at", STATIC_OR)
    return run_dir

def unzip_to(src_zip: Path, dest_dir: Path) -> List[Path]:
    out: List[Path] = []
    with zipfile.ZipFile(src_zip, "r") as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            target = dest_dir / Path(m.filename).name
            with zf.open(m) as inp, open(target, "wb") as outfp:
                outfp.write(inp.read())
            out.append(target)
    return out

def stage_upload_to_payload(zip_path: Path, run_dir: Path, case_id: str) -> List[Path]:
    if zip_path.is_dir():
        candidates = [p for p in zip_path.iterdir() if p.is_file()]
    else:
        tmp_incoming = run_dir / "_incoming"
        tmp_incoming.mkdir(exist_ok=True, parents=True)
        candidates = unzip_to(zip_path, tmp_incoming)

    out_docs: List[Path] = []
    dest_root = run_dir / "Payload documents" / case_id
    for p in candidates:
        if p.name.startswith("."):
            continue
        dest = dest_root / p.name
        shutil.copy2(p, dest)
        out_docs.append(dest)
    return out_docs

def slug(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80] or "doc"

def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf": return "application/pdf"
    if ext in (".xlsx", ".xls"): return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if ext == ".txt": return "text/plain"
    if ext == ".ifc": return "application/x-ifc"
    return "application/octet-stream"

def to_docspecs(staged: List[Path], case_id: str) -> List[DocSpec]:
    docs: List[DocSpec] = []
    base = f"https://example.org/{case_id}#"
    for i, p in enumerate(sorted(staged)):
        fn_rel = f"{case_id}/{p.name}"
        doc_iri = URIRef(f"{base}Doc_{i+1}")
        docs.append(
            DocSpec(
                iri=doc_iri,
                rel_filename=fn_rel,
                filetype=p.suffix.lower().lstrip(".") or "bin",
                format=_guess_mime(p),
                name=slug(p.stem),
            )
        )
    return docs

# ==== PDF/text preprocessing ==================================================
def _read_preview(path: Path, max_chars: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return f"[unreadable:{path.name}]"

def _extract_pdf_chunks(pdf_path: Path, max_chars_per_chunk: int = 1000) -> List[str]:
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
    except Exception:
        prev = _read_preview(pdf_path, max_chars=max_chars_per_chunk)
        return [prev] if prev else []

    chunks: List[str] = []
    for page_layout in extract_pages(str(pdf_path)):
        page_text_parts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text_parts.append(element.get_text())
        page_text = "\n".join(page_text_parts).strip()
        if not page_text:
            continue
        start = 0
        while start < len(page_text):
            end = min(len(page_text), start + max_chars_per_chunk)
            slice_ = page_text[start:end].strip()
            if slice_:
                chunks.append(slice_)
            start = end

    if not chunks:
        prev = _read_preview(pdf_path, max_chars=max_chars_per_chunk)
        if prev:
            chunks.append(prev)
    return chunks

def _guess_role_from_name(filename: str) -> Optional[str]:
    fn = filename.lower()
    if "application" in fn or "antrag" in fn: return "ApplicationForm"
    if "planning_statement" in fn or "planning statement" in fn: return "PlanningStatement"
    if "site" in fn and "plan" in fn: return "SitePlan"
    if "elevation" in fn: return "ElevationDrawing"
    if "floor" in fn and "plan" in fn: return "FloorPlan"
    return None

def preprocess_for_rag(case_id: str, run_dir: Path, docs: List[DocSpec], max_chars_per_chunk: int = 1000) -> Path:
    rag_dir = run_dir / "rag"
    chunks_path   = rag_dir / "chunks.jsonl"
    manifest_path = rag_dir / "manifest.json"

    manifest: Dict[str, Any] = {"case_id": case_id, "documents": []}

    with chunks_path.open("w", encoding="utf-8") as fout:
        for d in docs:
            staged = run_dir / "Payload documents" / case_id / Path(d.rel_filename).name
            if not staged.exists():
                print(f"[preprocess] WARNING: file missing: {staged}")
                continue

            role = _guess_role_from_name(staged.name)
            base_name = d.name
            ext = d.filetype.lower()

            if ext == "pdf":
                text_chunks = _extract_pdf_chunks(staged, max_chars_per_chunk)
            elif ext in {"txt", "csv", "xml", "json", "yaml", "yml", "ifc"}:
                text_chunks = [_read_preview(staged, max_chars=max_chars_per_chunk)]
            elif ext in {"xlsx", "xls"}:
                text_chunks = [_read_preview(staged, max_chars=max_chars_per_chunk)]
            else:
                text_chunks = [_read_preview(staged, max_chars=max_chars_per_chunk)]

            for i, txt in enumerate(text_chunks):
                tc = TextChunk(
                    case_id=case_id,
                    doc_iri=str(d.iri),
                    rel_filename=Path(d.rel_filename).name,
                    name=base_name,
                    role=role,
                    filetype=ext,
                    page=i + 1,
                    chunk_id=i,
                    text=txt or "",
                )
                fout.write(json.dumps(asdict(tc), ensure_ascii=False) + "\n")

            manifest["documents"].append(
                {
                    "doc_iri": str(d.iri),
                    "rel_filename": Path(d.rel_filename).name,
                    "name": base_name,
                    "role": role,
                    "filetype": ext,
                    "num_chunks": len(text_chunks),
                }
            )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preprocess] Wrote chunks to {chunks_path}")
    print(f"[preprocess] Wrote manifest to {manifest_path}")
    return rag_dir

# ==== Graph builders ==========================================================
def build_index_graph(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ct", CT); g.bind("ls", LS); g.bind("els", ELS)

    base = f"https://example.org/{case_id}#"
    container = URIRef(f"{base}Container_{case_id}")
    linkset   = URIRef(f"{base}Linkset_{case_id}")

    g.add((container, RDF.type, CT.ContainerDescription))
    g.add((linkset,   RDF.type, CT.Linkset))

    g.add((container, CT.conformanceIndicator, Literal("ICDD-Part1-Container", datatype=XSD.string)))
    g.add((container, CT.containsLinkset, linkset))

    publisher = URIRef(f"{base}Publisher_{case_id}")
    g.add((publisher, RDF.type, CT.Party))
    g.add((publisher, CT.name, Literal("icdd-rag-pipeline", datatype=XSD.string)))
    g.add((container, CT.publisher, publisher))

    for d in docs:
        g.add((d.iri, RDF.type, CT.Document))
        g.add((d.iri, RDF.type, CT.InternalDocument))
        g.add((d.iri, CT.belongsToContainer, container))
        g.add((d.iri, CT.filename, Literal(Path(d.rel_filename).name, datatype=XSD.string)))
        g.add((d.iri, CT.filetype, Literal(d.filetype, datatype=XSD.string)))
        g.add((d.iri, CT["format"], Literal(d.format, datatype=XSD.string)))
        g.add((d.iri, CT.name,     Literal(d.name, datatype=XSD.string)))
        g.add((container, CT.containsDocument, d.iri))

    g.add((linkset, CT.filename, Literal("Doc_Application_Links.rdf", datatype=XSD.string)))
    return g

def build_payload_linkset(case_id: str, docs: List[DocSpec]) -> Graph:
    g = Graph()
    g.bind("ls", LS); g.bind("els", ELS); g.bind("ct", CT)
    base = f"https://example.org/{case_id}#"

    if len(docs) >= 2:
        d1, d2 = docs[0], docs[1]
        link  = URIRef(f"{base}Link_1_{case_id}")
        fromE = URIRef(f"{base}Link_1_{case_id}#from")
        toE   = URIRef(f"{base}Link_1_{case_id}#to")
        idF   = URIRef(f"{base}Link_1_{case_id}#id_from")
        idT   = URIRef(f"{base}Link_1_{case_id}#id_to")
        g.add((link, RDF.type, LS.Link))
        g.add((link, LS.linkset, URIRef(f"{base}Linkset_{case_id}")))
        g.add((link, LS.type, Literal("DocDocLink", datatype=XSD.string)))

        g.add((fromE, RDF.type, LS.LinkElement))
        g.add((toE,   RDF.type, LS.LinkElement))
        g.add((idF,   RDF.type, LS.Identifier))
        g.add((idT,   RDF.type, LS.Identifier))

        g.add((fromE, LS.document, d1.iri))
        g.add((fromE, LS.hasIdentifier, idF))
        g.add((toE,   LS.document, d2.iri))
        g.add((toE,   LS.hasIdentifier, idT))
        g.add((idF,   LS.identifier, Literal("whole-doc", datatype=XSD.string)))
        g.add((idT,   LS.identifier, Literal("whole-doc", datatype=XSD.string)))

        g.add((link,  LS.hasFromLinkElement, fromE))
        g.add((link,  LS.hasToLinkElement,   toE))

    return g

# ==== Writers / Validators ====================================================
def write_files_and_zip(case_id: str, run_dir: Path, g_index: Graph, g_link: Graph) -> Path:
    (run_dir / "index.rdf").write_bytes(g_index.serialize(format="application/rdf+xml").encode("utf-8"))
    (run_dir / "Payload triples" / "Doc_Application_Links.rdf").write_bytes(
        g_link.serialize(format="application/rdf+xml").encode("utf-8")
    )

    out_zip = OUTPUT / f"ICDD_{case_id}.icdd"
    if out_zip.exists():
        out_zip.unlink()

    # Only include ISO 21597 parts: index.rdf + Payload documents + Payload triples + Ontology resources
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        include_roots = [
            run_dir / "index.rdf",
            run_dir / "Payload documents",
            run_dir / "Payload triples",
            run_dir / "Ontology resources",
        ]
        for root in include_roots:
            if root.is_file():
                z.write(root, arcname=str(root.relative_to(run_dir)))
            elif root.is_dir():
                for p in root.rglob("*"):
                    if not p.is_file():
                        continue
                    # Exclude .ipynb_checkpoints inside the container
                    if any(part.startswith(".ipynb_checkpoints") for part in p.parts):
                        continue
                    z.write(p, arcname=str(p.relative_to(run_dir)))
    return out_zip

def structural_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not (run_dir / "Payload triples" / "Doc_Application_Links.rdf").exists():
        errs.append("Payload file not found: Doc_Application_Links.rdf")
    need = ["Container.rdf", "Linkset.rdf", "ExtendedLinkset.rdf"]
    for n in need:
        if not (run_dir / "Ontology resources" / n).exists():
            errs.append(f"Missing ontology resource: {n}")
    return (len(errs) == 0, errs)

def _fmt(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".ttl", ".n3"): return "turtle"
    if ext in (".rdf", ".xml"): return "xml"
    return "turtle"

def coherence_check(run_dir: Path, case_id: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    try:
        g = Graph()
        g.parse(run_dir / "index.rdf", format="xml")
        for p in (run_dir / "Payload triples").glob("*.rdf"):
            g.parse(p, format="xml")
        # No deep logical checks here yet; just ensure parse OK
    except Exception as e:
        errs.append(f"RDF parse error: {e}")
    return (len(errs) == 0, errs)

def run_shacl_validation(run_dir: Path) -> Tuple[bool, str]:
    from pyshacl import validate

    data_graph = Graph()
    data_graph.parse(run_dir / "index.rdf", format="xml")
    for p in (run_dir / "Payload triples").glob("*.rdf"):
        data_graph.parse(p, format="xml")

    shapes_graph = Graph()
    candidates = [
        run_dir / "Ontology resources",
        run_dir / "Ontology resources" / "shapes",
        STATIC_OR,
    ]
    for base in candidates:
        if not base or not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".ttl", ".rdf", ".owl", ".xml"):
                try:
                    shapes_graph.parse(str(p), format=_fmt(p))
                except Exception:
                    pass

    conforms, rep_g, rep_t = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="rdfs",
        advanced=True,
        debug=False,
    )
    return (bool(conforms), rep_t)

# ==== MAIN ====================================================================
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)

    items = sorted([p for p in UPLOAD.iterdir() if not p.name.startswith(".")])
    if not items:
        print("No submissions found in", UPLOAD)
        return

    print("Found submissions:")
    for s in items:
        print(" -", s)

    for item in items:
        case_id = make_case_id(item)
        print(f"\n=== Processing {item.name} → CASE_ID={case_id} ===")

        run_dir = ensure_run_dir(case_id)
        staged_abs = stage_upload_to_payload(item, run_dir, case_id)
        if not staged_abs:
            print("No files discovered; skipping.")
            continue

        docs = to_docspecs(staged_abs, case_id)
        g_index = build_index_graph(case_id, docs)
        g_link  = build_payload_linkset(case_id, docs)

        # Preprocessing for RAG
        rag_dir = preprocess_for_rag(case_id, run_dir, docs, max_chars_per_chunk=1000)

        # LightRAG index
        try:
            build_lightrag_index(case_id, rag_dir)
        except Exception as e:
            print(f"[LightRAG] index build failed: {e}")

        # OntoBPR via LLM (this is now strictly required)
        try:
            data = extract_schema_with_llm(case_id, rag_dir)
            base = f"https://example.org/{case_id}#"
            container_iri = URIRef(f"{base}Container_{case_id}")
            g_onto = schema_to_ontobpr(case_id, container_iri, data)
            onto_path = run_dir / "Payload triples" / "OntoBPR.ttl"
            onto_path.write_text(g_onto.serialize(format="turtle"), encoding="utf-8")
            print("[OntoBPR-LLM] wrote OntoBPR.ttl:", onto_path)
        except Exception as e:
            print("[OntoBPR-LLM] extraction failed (fatal):", e)
            # Do NOT continue if OntoBPR is missing
            raise

        icdd_path = write_files_and_zip(case_id, run_dir, g_index, g_link)
        print("✅ ICDD written:", icdd_path)

        ok, errs = structural_check(run_dir, case_id)
        print("=== Structural check:", "OK" if ok else "FAIL", "===")
        for e in errs:
            print(" -", e)

        ok2, errs2 = coherence_check(run_dir, case_id)
        print("=== Coherence check:", "OK" if ok2 else "FAIL", "===")
        for e in errs2:
            print(" -", e)

        try:
            conf, report = run_shacl_validation(run_dir)
            print("\n=== SHACL RESULT ===")
            print("Conforms:", conf)
            print(report)
        except Exception as e:
            print("(SHACL validation skipped:", e, ")")

    print("\nAll submissions processed.")

if __name__ == "__main__":
    main()
