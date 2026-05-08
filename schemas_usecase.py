from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union
from enum import Enum

# =========================
# SHARED CANONICAL TYPES
# =========================

SepsisVariable = Literal[
    "mortality_90d", "mortality_28d", "mortality_hospital", "mortality_icu",
    "sofa_score", "apache_ii_score",
    "antibiotic_time_to_admin", "fluid_volume_6h",
    "vasopressor_use", "vasopressor_duration", "lactate",
    "il6_level", "lymphocyte_count",  # Added for Use Case 1
    "procalcitonin", "crp",
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
    AUC_ROC = "auc_roc"        # Added for Prognostic accuracy
    RISK_DIFF = "risk_difference"
    COUNT = "count"
    OTHER = "other"

# =========================
# PASS 1 (ATOMIC EXTRACTION - UPGRADED FOR USE CASE 1)
# =========================

class AtomicExtraction(BaseModel):
    variable: str = Field(..., description="The clinical parameter or biomarker")
    value: Optional[str] = Field(None, description="The primary numerical value or percentage")
    
    # --- PROGNOSTIC FIELDS FOR COUNTERFACTUALS ---
    effect_size_type: Optional[MetricType] = Field(None, description="OR, HR, or AUC if this is a predictive association")
    ci_lower: Optional[float] = Field(None, description="Lower bound of 95% Confidence Interval")
    ci_upper: Optional[float] = Field(None, description="Upper bound of 95% Confidence Interval")
    p_value: Optional[str] = Field(None, description="Statistical significance (e.g., p < 0.05)")
    
    outcome_linked: Optional[str] = Field(None, description="The outcome this variable predicts (e.g., '28-day mortality')")
    # ---------------------------------------------
    
    unit: Optional[str] = None
    raw_value_text: Optional[str] = Field(default="N/A")
    evidence_text: Optional[str] = Field(default="N/A", description="The exact sentence from the PDF")
    source_type: Optional[str] = Field(default="text")

class PaperExtractions(BaseModel):
    paper_id: str
    title: Optional[str] = None
    year: Optional[Union[str, int]] = None
    extractions: List[AtomicExtraction]

# =========================
# PASS 2 (ENRICHED BENCHMARKING)
# =========================

class EnrichedClaim(BaseModel):
    """
    Final representation for Use Case 1 structured evidence table.
    """
    variable: SepsisVariable
    intent: Literal["observed_outcome", "threshold", "baseline", "prognostic_association"]
    
    # Statistical modeling fields
    value: Optional[float] = None
    effect_size: Optional[float] = None
    metric: MetricType
    
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[str] = None
    
    # Counterfactual Context
    target_outcome: Optional[str] = Field(None, description="e.g. 28-day mortality")
    population_severity: Optional[str] = Field(None, description="Mean SOFA/APACHE of the study cohort to allow matching")
    
    reasoning: str = Field(..., description="AI explanation of the prognostic link")
    supporting_spans: List[str] # List of evidence_text strings