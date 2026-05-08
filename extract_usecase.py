import os
import instructor
from openai import OpenAI
from dotenv import load_dotenv
import json
from pathlib import Path
from schemas import PaperExtractions # Ensure your schema includes effect_size, outcome, etc.

load_dotenv()

# Setup cache directory for Use Case 1
CACHE_DIR = Path("data/mortality_counterfactuals")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- OPENROUTER CONFIGURATION ---
client = instructor.from_openai(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"), # Add this to your .env
    ),
    mode=instructor.Mode.JSON,
)

# You can now use more powerful models for better extraction
# Examples: "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini", or "meta-llama/llama-3.1-70b"
CHOSEN_MODEL = "anthropic/claude-3.5-sonnet" 

# Target variables specifically tuned for Counterfactual Mortality Estimation
TARGET_VARIABLES = """
- PREDICTORS: baseline lactate, IL-6, SOFA score, APACHE II, Age, Lymphocyte count.
- STATISTICAL PARAMETERS: Odds Ratio (OR), Hazard Ratio (HR), Area Under Curve (AUC).
- CONFIDENCE INTERVALS: 95% CI (lower and upper bounds).
- OUTCOME DEFINITIONS: (e.g., 28-day mortality vs. in-hospital mortality).
- COHORT BASELINE: Mean/Median SOFA score of the study group (needed to match your registry).
"""

def build_counterfactual_prompt(paper) -> str:
    # Filter for Results and Methods where statistical modeling occurs
    relevant_text = "\n\n".join([
        f"## {s.heading}\n{s.text}" 
        for s in paper.sections 
        if any(kw in s.heading.lower() for kw in ["result", "statistical", "prognostic", "outcome"])
    ])
    
    return f"""TASK: Extract statistical data for counterfactual mortality benchmarking.
Paper: {paper.paper_id}

### EXTRACTION RULES:
1. Identify clinical variables associated with mortality.
2. Extract the Effect Size: We need the OR, HR, or AUC to calculate risk.
3. Extract the 95% Confidence Interval for each effect size.
4. Define the outcome (e.g., 28-day mortality).
5. Document the Cohort Baseline: What was the average severity (SOFA/APACHE) of the patients?

### TARGET VARIABLES:
{TARGET_VARIABLES}

### TEXT CONTENT:
{relevant_text[:8000]} 

FINAL CHECK: Ensure you capture the statistical relationship (e.g., 'For every 1pt increase in SOFA, mortality OR was 1.2').
"""

def extract_mortality_evidence(paper, force_refresh: bool = False):
    cache_path = CACHE_DIR / f"{paper.paper_id}_mortality.json"
    
    if cache_path.exists() and not force_refresh:
        with open(cache_path, "r") as f:
            return json.load(f)
    
    # Using OpenRouter to handle the extraction
    result = client.chat.completions.create(
        model=CHOSEN_MODEL,
        response_model=PaperExtractions,
        max_retries=2,
        messages=[
            {
                "role": "system", 
                "content": "You are a senior clinical data scientist extracting prognostic evidence from sepsis literature."
            },
            {"role": "user", "content": build_counterfactual_prompt(paper)}
        ],
        # OpenRouter specific headers (Optional)
        extra_headers={
            "HTTP-Referer": "https://your-project-url.com", 
            "X-Title": "Sepsis Hackathon Extraction",
        }
    )
    
    with open(cache_path, "w") as f:
        f.write(result.model_dump_json(indent=2))
    return result