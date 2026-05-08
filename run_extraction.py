import os
import json
import instructor
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel # We use this to make a fake Section

# 1. IMPORT ONLY WHAT WORKS
from ingest import ParsedPaper 
from schemas import PaperExtractions 

load_dotenv()

# 2. DEFINE A LOCAL SECTION CLASS 
# This mimics what ParsedPaper expects
class LocalSection(BaseModel):
    heading: str
    text: str

# 3. OPENROUTER CONFIG
client = instructor.from_openai(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
    mode=instructor.Mode.JSON,
)

INPUT_DIR = Path("./data/parsed_papers") 
OUTPUT_DIR = Path("./data/mortality_counterfactuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_mortality_evidence(paper):
    # We combine all section text for the AI
    content = " ".join([s.text for s in paper.sections])
    
    prompt = f"""
    TASK: Extract statistical associations for counterfactual mortality modeling.
    Paper: {paper.paper_id}
    Content: {content[:10000]} 
    """
    
    return client.chat.completions.create(
        model="openai/gpt-4o-mini",
        response_model=PaperExtractions,
        messages=[
            {"role": "system", "content": "You are a clinical data extractor."},
            {"role": "user", "content": prompt}
        ]
    )

def main():
    files = list(INPUT_DIR.glob("*.json"))
    for file_path in files:
        print(f"📄 Processing: {file_path.name}")
        with open(file_path, "r") as f:
            paper_data = json.load(f)
            
            try:
                # If it's a list, we wrap the dicts into our LocalSection class
                if isinstance(paper_data, list):
                    cleaned_sections = [
                        LocalSection(
                            heading=item.get("heading", "Section"), 
                            text=item.get("text", str(item))
                        ) for item in paper_data
                    ]
                    
                    # We bypass the strict constructor by using .construct() 
                    # or just passing the list if ParsedPaper allows it
                    paper_obj = ParsedPaper(
                        paper_id=file_path.stem, 
                        sections=cleaned_sections,
                        tables=[],
                        figures=[],
                        full_markdown=""
                    )
                else:
                    paper_obj = ParsedPaper(**paper_data)
                
                result = extract_mortality_evidence(paper_obj)
                
                out_path = OUTPUT_DIR / f"{paper_obj.paper_id}_results.json"
                with open(out_path, "w") as f:
                    f.write(result.model_dump_json(indent=2))
                print(f"✅ Saved to {out_path}")

            except Exception as e:
                print(f"⚠️ Error with {file_path.name}: {e}")

if __name__ == "__main__":
    main()