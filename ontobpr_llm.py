# ontobpr_llm.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from field_schema import default_schema
from llm_backend import call_llm

ONTOBPR = Namespace("https://w3id.org/ontobpr#")
CT      = Namespace("https://standards.iso.org/iso/21597/-1/ed-1/en/Container#")

# -------------------------------------------------------------------
# Optional LightRAG context (best-effort, never fatal)
# -------------------------------------------------------------------
try:
    from rag_lightrag import retrieve_context_for_case  # type: ignore
    _HAS_LIGHTRAG = True
except Exception as e:
    print("[ontobpr_llm] LightRAG retrieval unavailable:", e)
    _HAS_LIGHTRAG = False


def _load_chunks_for_case(rag_dir: Path) -> List[Dict[str, Any]]:
    """Load JSONL chunks produced by preprocess_for_rag()."""
    chunks_path = rag_dir / "chunks.jsonl"
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def extract_schema_with_llm(case_id: str, rag_dir: Path) -> Dict[str, Any]:
    """
    Main semantic extraction: fill the default schema using an LLM + (optionally) LightRAG context.
    Returns a dict matching the JSON schema (application + building).
    """
    schema = default_schema()
    chunks = _load_chunks_for_case(rag_dir)

    # ---------------------------------------------------------------
    # 1) Base context from chunks (ApplicationForm / PlanningStatement first)
    # ---------------------------------------------------------------
    application_texts = [
        c["text"]
        for c in chunks
        if "ApplicationForm" in (c.get("role") or "")
        or "PlanningStatement" in (c.get("role") or "")
        or "planning statement" in (c.get("name") or "").lower()
    ]
    if not application_texts:
        application_texts = [c["text"] for c in chunks]

    base_text = "\n\n".join(application_texts[:40])  # limit context size

    # ---------------------------------------------------------------
    # 2) Optional LightRAG context (best-effort only)
    # ---------------------------------------------------------------
    rag_context = ""
    if _HAS_LIGHTRAG:
        try:
            # NOTE: positional args to avoid 'rag_dir=' mismatch
            rag_context = retrieve_context_for_case(case_id, rag_dir)
        except Exception as e:
            print(f"[ontobpr_llm] LightRAG retrieval failed ({e}); continuing with chunks only.")

    combined_text = base_text
    if rag_context:
        combined_text = base_text + "\n\n[RETRIEVED CONTEXT]\n" + rag_context

    # ---------------------------------------------------------------
    # 3) JSON template for the LLM
    # ---------------------------------------------------------------
    json_template: Dict[str, Any] = {
        "case_id": case_id,
        "building_application": {f.id: None for f in schema.application_fields},
        "building": {f.id: None for f in schema.building_fields},
    }

    # Compose field descriptions
    field_desc_lines: List[str] = []
    for f in schema.application_fields + schema.building_fields:
        desc = f"- {f.id} ({f.answer_type}"
        if f.enum_values:
            desc += f", one of: {', '.join(f.enum_values)}"
        desc += f") – " + f.question
        field_desc_lines.append(desc)

    user_prompt = f"""
You are extracting structured data from a building planning application.

Below is text from the planning application form and planning statement
for a single case (case_id={case_id}), possibly enriched with retrieved context.
Your job is to fill a JSON object with specific fields.

IMPORTANT RULES:
- Only use information that is explicitly stated in the text.
- If a field is not mentioned or cannot be determined, set its value to null.
- For yes/no fields, answer with true, false, or null (for unknown).
- For dates, use ISO format YYYY-MM-DD if possible, otherwise null.
- For numbers, use a numeric value if stated, otherwise null.

FIELDS TO EXTRACT:
{chr(10).join(field_desc_lines)}

Return ONLY valid JSON, matching exactly this template (keys and structure):

{json.dumps(json_template, indent=2)}

TEXT:
\"\"\"
{combined_text}
\"\"\"
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise information extraction assistant for building permit applications. "
                "You must respond with valid JSON only, matching the provided template exactly."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # ---------------------------------------------------------------
    # 4) Call LLM via llm_backend (NO extra kwargs!)
    # ---------------------------------------------------------------
    raw = call_llm(messages)  # <— IMPORTANT: no max_new_tokens here

    # ---------------------------------------------------------------
    # 5) Parse JSON with robust fallback
    # ---------------------------------------------------------------
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end + 1])
        else:
            raise ValueError(f"LLM did not return valid JSON: {raw[:200]}...")

    return data


def schema_to_ontobpr(case_id: str, container_iri: URIRef, data: Dict[str, Any]) -> Graph:
    """
    Turn the extracted schema into OntoBPR triples.
    """
    g = Graph()
    g.bind("ontobpr", ONTOBPR)
    g.bind("ct", CT)

    base = f"https://example.org/{case_id}#"
    ba   = URIRef(f"{base}BA_{case_id}")
    bldg = URIRef(f"{base}Building_{case_id}")

    # BuildingApplication
    g.add((ba, RDF.type, ONTOBPR.BuildingApplication))
    g.add((ba, ONTOBPR.hasBuildingApplicationContainer, container_iri))

    app = data.get("building_application", {})

    ref = app.get("application_reference")
    if ref:
        g.add((ba, ONTOBPR.applicationIdentifier, Literal(ref, datatype=XSD.string)))

    app_type = app.get("application_type")
    if app_type:
        g.add((ba, ONTOBPR.applicationType, Literal(app_type, datatype=XSD.string)))

    is_retro = app.get("is_retrospective")
    if isinstance(is_retro, bool):
        g.add((ba, ONTOBPR.isRetrospective, Literal(is_retro, datatype=XSD.boolean)))

    auth = app.get("authority_name")
    if auth:
        g.add((ba, ONTOBPR.buildingAuthorityName, Literal(auth, datatype=XSD.string)))

    addr = app.get("site_address")
    if addr:
        g.add((ba, ONTOBPR.siteAddress, Literal(addr, datatype=XSD.string)))

    # Building
    g.add((bldg, RDF.type, ONTOBPR.Building))
    b = data.get("building", {})
    use = b.get("building_use_type")
    if use:
        g.add((bldg, ONTOBPR.buildingUseType, Literal(use, datatype=XSD.string)))

    storeys = b.get("number_of_storeys")
    if isinstance(storeys, (int, float)):
        g.add((bldg, ONTOBPR.numberOfStoreys, Literal(int(storeys), datatype=XSD.integer)))

    # Link BuildingApplication ↔ Building
    g.add((ba, ONTOBPR.hasBuilding, bldg))

    return g
