import json
import os
import time
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INPUT_FILE  = "schemes.json"
OUTPUT_FILE = "schemes_cleaned.json"
BATCH_SAVE_EVERY = 50   # save progress every 50 schemes

print("Loading schemes...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    schemes = json.load(f)
print(f"Loaded {len(schemes)} schemes")

processed = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        processed = json.load(f)
    print(f"Resuming from scheme {len(processed)} (already processed)")

already_done = len(processed)
remaining = schemes[already_done:]
print(f"Schemes remaining to process: {len(remaining)}")

# clean benefits text using Groq
def clean_benefits(raw_text: str, scheme_name: str) -> str:
    """
    Send messy benefits text to Groq and get back clean structured text.
    """
    if not raw_text or len(raw_text.strip()) < 20:
        return raw_text

    prompt = f"""You are a data cleaning assistant. Clean and restructure the following government scheme benefits text.

Scheme: {scheme_name}

Raw benefits text:
{raw_text[:800]}

Instructions:
- Extract all benefit amounts and what they are for
- Format as: "Category: Amount | Category: Amount" using pipe separators
- If there's a table with different amounts for different courses/categories, format each row clearly like: "Course/Category Name: Rs.X"
- Keep all important numbers and conditions
- Remove formatting artifacts like characters
- Keep it concise but complete
- Do NOT add any information that is not in the original text
- Respond with ONLY the cleaned benefits text, no explanation

Example output format:
"Tuition Fee Grant: Medical/MBBS: Rs.2,00,000 | B.Tech/BE/Professional: Rs.50,000 | Diploma: Rs.25,000 | BA/BSc/BCom: Rs.10,000. Hostel Grant: Rs.1,200/month. Books Grant: Engineering: Rs.5,000 | Diploma: Rs.3,000."
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        cleaned = response.choices[0].message.content.strip()
        # Remove any quotes if LLM wrapped in quotes
        cleaned = cleaned.strip('"').strip("'")
        return cleaned
    except Exception as e:
        print(f"  Error cleaning benefits: {e}")
        return raw_text  # return original if error

# Helper: clean eligibility text 
def clean_eligibility(raw_text: str) -> str:
    """
    Clean eligibility text into clear bullet-point style.
    """
    if not raw_text or len(raw_text.strip()) < 20:
        return raw_text

    prompt = f"""Clean and restructure this government scheme eligibility text into clear, simple conditions.

Raw eligibility text:
{raw_text[:600]}

Instructions:
- List each eligibility condition clearly
- Format as numbered list: "1. Condition | 2. Condition | 3. Condition"
- Keep all important criteria (age, income, state, caste, occupation)
- Remove formatting artifacts
- Keep it concise
- Respond with ONLY the cleaned eligibility text, no explanation
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        cleaned = response.choices[0].message.content.strip()
        cleaned = cleaned.strip('"').strip("'")
        return cleaned
    except Exception as e:
        return raw_text

# Main processing loop 
print(f"\nStarting cleaning with Groq (this will take ~30-40 mins for 3400 schemes)...")
print("Script auto-saves every 50 schemes. Safe to interrupt and resume.\n")

for i, scheme in enumerate(tqdm(remaining, desc="Cleaning")):
    scheme_name = scheme.get("scheme_name", "")

    # Clean benefits
    raw_benefits = scheme.get("benefits", "")
    if raw_benefits and len(raw_benefits) > 50:
        scheme["benefits"] = clean_benefits(raw_benefits, scheme_name)

    # Clean eligibility_raw
    raw_eligibility = scheme.get("eligibility_raw", "")
    if raw_eligibility and len(raw_eligibility) > 50:
        scheme["eligibility_raw"] = clean_eligibility(raw_eligibility)

    processed.append(scheme)

    # Save progress every BATCH_SAVE_EVERY schemes
    if (i + 1) % BATCH_SAVE_EVERY == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        print(f"\n  Progress saved: {len(processed)}/{len(schemes)} schemes done")

    # Rate limit: Groq allows ~30 req/min
    # 2 calls per scheme = 60 calls per 30 schemes = need ~1s delay
    time.sleep(0.5)

# Final save 
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)


print(f"\nDone! {len(processed)} schemes cleaned and saved to {OUTPUT_FILE}")
