import os
import requests
import time

def download_pmc_pdf(pmcid):
    """
    Directly targets the download link used by the browser.
    """
    # Standard PMC PDF link structure
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
    
    # We must pretend to be a browser (User-Agent) or NCBI will block the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"Attempting download for PMC{pmcid}...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200 and b'%PDF' in response.content[:10]:
            with open(f"PMC{pmcid}.pdf", "wb") as f:
                f.write(response.content)
            print(f"✅ Success: PMC{pmcid}.pdf saved.")
            return True
        else:
            print(f"❌ Failed: PMC{pmcid} (Status: {response.status_code}). Might not be Open Access.")
            return False
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return False

# Test with a known Open Access Sepsis paper (PMC10283084)
download_pmc_pdf("10283084")
