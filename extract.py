import os
import instructor
from openai import OpenAI
from dotenv import load_dotenv
from schemas import PaperClaims
from ingest import ParsedPaper
import json
from pathlib import Path
from schemas import PaperExtractions, AtomicExtraction
load_dotenv()

# Setup cache directory
CACHE_DIR = Path("data/extracted")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- OLLAMA LOCAL CONFIGURATION ---
# Points to your local machine instead of the cloud.
# The API key is a placeholder (Ollama doesn't require one).
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ENTER API KEY HERE (NOT USED FOR OLLAMA)", 
    ),
    mode=instructor.Mode.JSON, # Crucial for structured output from local models
)

# Use llama3.2 (3B) to stay within 4GB VRAM limits
LOCAL_MODEL = "llama3.2" 

TARGET_VARIABLES = """
- mortality_90d, mortality_28d, mortality_hospital, mortality_icu
- sofa_score
- apache_ii_score
- antibiotic_time_to_admin
- fluid_volume_6h
- vasopressor_use: Percentage of patients on vasopressors or a binary indicator. (Do NOT confuse with MAP/blood pressure targets), vasopressor_duration
- lactate
- mechanical_ventilation_use, mechanical_ventilation_duration
- renal_replacement_therapy_use
- length_of_stay_icu, length_of_stay_hospital
- sample_size, age, sex_male_percent
"""

RELEVANT_SECTIONS = {
    "abstract", "background", "results", "methods",
    "outcomes", "primary outcome", "secondary outcome",
    "patient characteristics", "study patients",
    "interventions", "physiological"
}

def filter_sections(paper: ParsedPaper) -> str:
    filtered = [
        f"## {s.heading}\n{s.text}"
        for s in paper.sections
        if any(kw in s.heading.lower() for kw in RELEVANT_SECTIONS)
    ]
    # Local models have smaller context windows; we must be aggressive
    if len(filtered) < 3:
        filtered = [f"## {s.heading}\n{s.text}" for s in paper.sections[:4]]
    return "\n\n".join(filtered)

def build_prompt(paper: ParsedPaper) -> str:
    sections_text = filter_sections(paper)
    
    return f"""TASK: Extract EVERY clinical value related to Sepsis. 
Paper: {paper.paper_id}

### EXTRACTION RULES:
1. Look for BASELINE data (Age, Sample Size, Sex).
2. Look for CLINICAL thresholds (SOFA >= 2, Lactate > 2).
3. Look for OUTCOMES (Mortality %, ICU Length of Stay).
4. For every value, you MUST provide the 'evidence_text' (the sentence it came from).

### TARGET VARIABLES:
{TARGET_VARIABLES}

### OUTPUT FORMAT (MANDATORY):
{{
  "paper_id": "{paper.paper_id}",
  "title": "...",
  "year": "...",
  "extractions": [
    {{ "variable": "sample_size", "value": "155", ... }},
    {{ "variable": "age", "value": "65", ... }},
    {{ "variable": "mortality_hospital", "value": "10%", ... }}
  ]
}}

### TEXT CONTENT:
{sections_text[:5000]}

FINAL CHECK: Did you find at least 5-10 extractions? If they are in the text, you must include them.
"""

def extract_claims(paper: ParsedPaper, force_refresh: bool = False) -> PaperExtractions:
    cache_path = CACHE_DIR / f"{paper.paper_id}_raw.json"
    
    if cache_path.exists() and not force_refresh:
        with open(cache_path, "r") as f:
            return PaperExtractions.model_validate_json(f.read())
    
    # We use a slightly more aggressive retry and a lower temperature
    result = client.chat.completions.create(
        model="llama3.2",
        response_model=PaperExtractions,
        max_retries=3,
        temperature=0.0, # CRITICAL: 0.0 makes the model more predictable/robotic
        max_tokens=3000,
        messages=[
            {"role": "system", "content": "You are a precise medical data extractor. Extract ONLY values explicitly stated in the text. If a variable is not found, omit it. NEVER invent or infer values not present in the text."},
            {"role": "user", "content": build_prompt(paper)}
        ]
    )
    #Store the raw extraction in the cache_path
    with open(cache_path, "w") as f:
        f.write(result.model_dump_json(indent=2))
    return result

