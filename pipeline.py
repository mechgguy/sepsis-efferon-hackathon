# test_extract.py
from ingest import parse_pdf
from extract import extract_claims

paper = parse_pdf("papers/The ARISE Investigators and the ANZICS Clinical Trials Group - 2014 - Goal-Directed Resuscitation for Patients with Early Septic Shock.pdf")
# This now returns a PaperExtractions object (Pass 1)
raw_data = extract_claims(paper)

print(f"Paper ID: {raw_data.paper_id}")
print(f"Year: {raw_data.year}")
# Change 'claims' to 'extractions' to match the schema
print(f"Atomic Extractions found: {len(raw_data.extractions)}\n")

for e in raw_data.extractions:
    # Use the field names from your new AtomicExtraction class
    print(f"[{e.variable}] Value: {e.value or 'N/A'} {e.unit or ''}")
    print(f"  Raw Text: {e.raw_value_text}")
    print(f"  Evidence: {(e.evidence_text or 'N/A')[:100]}...")
    print("-" * 30)