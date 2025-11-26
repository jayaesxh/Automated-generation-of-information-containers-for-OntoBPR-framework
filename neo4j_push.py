# neo4j_push.py
from __future__ import annotations

import os
import json
from pathlib import Path

from neo4j import GraphDatabase
from rdflib import Graph, Namespace, RDF

ONTOBPR = Namespace("https://w3id.org/ontobpr#")


def push_case_to_neo4j(case_id: str, run_dir: Path) -> None:
    """
    Push OntoBPR + document info for case_id into Neo4j.

    Requires env vars:
      NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
      optional: NEO4J_DB (default 'neo4j')
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DB", "neo4j")

    if not (uri and user and password):
        print("[Neo4j] Credentials missing; skipping push.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))

    onto_path = run_dir / "Payload triples" / "OntoBPR.ttl"
    g = Graph()
    if onto_path.exists():
        g.parse(str(onto_path), format="turtle")
    else:
        print("[Neo4j] OntoBPR.ttl not found; only documents will be pushed.")

    manifest_path = run_dir / "rag" / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with driver.session(database=db) as session:
        # Clear existing nodes for this case
        session.run("MATCH (n {case_id:$case_id}) DETACH DELETE n", case_id=case_id)

        # BuildingApplication nodes
        for ba in g.subjects(RDF.type, ONTOBPR.BuildingApplication):
            app_id = None
            addr = None
            for _, _, v in g.triples((ba, ONTOBPR.applicationIdentifier, None)):
                app_id = str(v)
            for _, _, v in g.triples((ba, ONTOBPR.siteAddress, None)):
                addr = str(v)

            session.run(
                """
                MERGE (ba:BuildingApplication:OntoBPR {iri:$iri})
                SET ba.case_id=$case_id,
                    ba.applicationId=$app_id,
                    ba.siteAddress=$addr
                """,
                iri=str(ba),
                case_id=case_id,
                app_id=app_id,
                addr=addr,
            )

        # Document nodes from manifest
        docs = manifest.get("documents", [])
        for d in docs:
            doc_iri = d.get("doc_iri")
            name = d.get("name")
            role = d.get("role")
            filetype = d.get("filetype")

            session.run(
                """
                MERGE (doc:Document {iri:$iri})
                SET doc.case_id=$case_id,
                    doc.name=$name,
                    doc.role=$role,
                    doc.filetype=$filetype
                """,
                iri=doc_iri,
                case_id=case_id,
                name=name,
                role=role,
                filetype=filetype,
            )

            session.run(
                """
                MATCH (ba:BuildingApplication {case_id:$case_id}),
                      (doc:Document {iri:$iri})
                MERGE (ba)-[:HAS_DOCUMENT]->(doc)
                """,
                case_id=case_id,
                iri=doc_iri,
            )

    driver.close()
    print(f"[Neo4j] Push complete for case {case_id}.")
