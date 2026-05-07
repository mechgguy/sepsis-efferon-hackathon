
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from enum import Enum
from typing import Optional, Union

# =========================
# SHARED CANONICAL TYPES
# =========================

SepsisVariable = Literal[
    "mortality_90d", "mortality_28d", "mortality_hospital", "mortality_icu",
    "sofa_score", "apache_ii_score",
    "antibiotic_time_to_admin", "fluid_volume_6h",
    "vasopressor_use", "vasopressor_duration", "lactate",
    "mechanical_ventilation_use", "mechanical_ventilation_duration",
    "renal_replacement_therapy_use",
    "length_of_stay_icu", "length_of_stay_hospital",
    "sample_size", "age", "sex_male_percent"
]


class MetricType(str, Enum):
    MEAN_SD = "mean_sd"
    MEDIAN_IQR = "median_iqr"
    PROPORTION = "proportion"
    HAZARD_RATIO = "hazard_ratio"
    ODDS_RATIO = "odds_ratio"
    RISK_DIFF = "risk_difference"
    COUNT = "count"
    OTHER = "other"


class ClaimIntent(str, Enum):
    OBSERVED_OUTCOME = "observed_outcome"
    DEFINITIONAL_THRESHOLD = "threshold"
    BASELINE_CHARACTERISTIC = "baseline"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =========================
# PASS 1 (ATOMIC EXTRACTION)
# =========================

class RawSpan(BaseModel):
    """
    Grounding unit. Always required.
    """
    text: str = Field(..., min_length=15, max_length=300)
    section: Literal["abstract", "methods", "results", "discussion", "other"]
    source_type: Literal["text", "table"]
    page: Optional[int] = None


class NumericValue(BaseModel):
    """
    Pure numeric extraction. No interpretation.
    """
    value: Optional[float] = None
    value_secondary: Optional[float] = None
    unit: Optional[str] = None
    raw_text: str = Field(..., min_length=1)


class Context(BaseModel):
    """
    Light context only. No normalization required.
    """
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    time_point: Optional[str] = None


# =========================
# PASS 1 (ATOMIC EXTRACTION - 3B VERSION)
# =========================

class AtomicExtraction(BaseModel):
    variable: str
    value: Optional[str] = None
    unit: Optional[str] = None
    raw_value_text: Optional[str] = Field(default="N/A")   # allow None
    evidence_text: Optional[str] = Field(default="N/A")    # allow None
    source_type: Optional[str] = Field(default="text")

class PaperExtractions(BaseModel):
    paper_id: str
    title: Optional[str] = None
    year: Optional[Union[str, int]] = None   # accept both
    extractions: List[AtomicExtraction]
# =========================
# PASS 2 (ENRICHMENT / REASONING)
# =========================

class EnrichedClaim(BaseModel):
    """
    Fully interpreted, validated claim.
    Produced ONLY in pass 2.
    """
    id: str  # inherited from AtomicExtraction

    variable: SepsisVariable
    intent: ClaimIntent

    # parsed statistics
    value: Optional[float] = None
    value_secondary: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[str] = None

    unit: Optional[str] = None
    statistical_format: Optional[MetricType] = None

    # normalized context
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    time_point: Optional[str] = None

    # provenance (multiple spans allowed after merging)
    supporting_spans: List[RawSpan]

    confidence: Confidence
    reasoning: str = Field(
        ...,
        description="Short explanation of how the claim was derived from supporting spans"
    )


class PaperClaims(BaseModel):
    """
    Final per-paper structured output after pass 2.
    """
    paper_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    study_design: Literal[
        "RCT", "observational", "systematic_review", "meta_analysis", "other"
    ]

    sample_size: Optional[int] = None

    claims: List[EnrichedClaim]


# =========================
# OPTIONAL: CROSS-PAPER AGGREGATION
# =========================

class AggregatedClaim(BaseModel):
    """
    Cross-paper synthesis (optional stage).
    """
    normalized_claim: str

    variable: SepsisVariable

    supporting_papers: List[str]
    conflicting_papers: List[str]

    confidence: Confidence

    reasoning: str
