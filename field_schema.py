# field_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any

AnswerType = Literal["string", "bool", "date", "number", "enum"]

@dataclass
class FieldSpec:
    id: str
    question: str           # natural language prompt to the LLM
    answer_type: AnswerType
    enum_values: Optional[List[str]] = None
    required: bool = False

@dataclass
class ExtractionSchema:
    application_fields: List[FieldSpec]
    building_fields: List[FieldSpec]

def default_schema() -> ExtractionSchema:
    """
    Minimal but meaningful schema you can map to OntoBPR.
    Extend later if you want more properties.
    """
    return ExtractionSchema(
        application_fields=[
            FieldSpec(
                id="application_reference",
                question="What is the planning application reference number?",
                answer_type="string",
                required=True,
            ),
            FieldSpec(
                id="application_type",
                question="What is the type of planning application (e.g. full planning, variation of condition, listed building)?",
                answer_type="string",
                required=True,
            ),
            FieldSpec(
                id="is_retrospective",
                question="Is the application retrospective (yes/no)?",
                answer_type="bool",
                required=False,
            ),
            FieldSpec(
                id="applicant_name",
                question="What is the applicant's or applicant company’s name?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="authority_name",
                question="Which local planning authority is this application submitted to?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="site_address",
                question="What is the site address (street, town/city, postcode)?",
                answer_type="string",
                required=True,
            ),
        ],
        building_fields=[
            FieldSpec(
                id="building_use_type",
                question="What is the primary current or proposed use of the building?",
                answer_type="string",
                required=False,
            ),
            FieldSpec(
                id="number_of_storeys",
                question="How many storeys does the building have (if stated)?",
                answer_type="number",
                required=False,
            ),
        ],
    )
