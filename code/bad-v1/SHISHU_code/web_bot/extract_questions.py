import glob, json, os, time, random
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODELS = ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"] # main model, then lighter backup

text = "\n\n=== NEXT CALL ===\n\n".join(
    open(f, encoding="utf-8").read() for f in glob.glob("transcripts/*.txt"))

prompt = f"""Below are transcripts of calls between a clinic and mothers/guardians.
Extract every question the clinic asked that helps understand a child's/mother's health issue.
Merge duplicates, rewrite each as a short clear question, and group by category
(e.g. basic info, symptoms, duration, feeding, medication, history).
Return ONLY a JSON list: [{{"id":"q1","category":"...","question":"...","required":true}}]

{text}"""

r = None
for attempt in range(8):
    model = MODELS[0] if attempt < 4 else MODELS[1]
    try:
        print(f"Attempt {attempt+1} with {model}...")
        r = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        break
    except Exception as e:
        print("  failed:", str(e)[:150])
        time.sleep(10 * (attempt + 1) + random.random())   # waits 10s, 20s, 30s...

if r is None:
    raise SystemExit("Google is still busy. Try again in 10-15 minutes.")

json.dump(json.loads(r.text), open("questions_2.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Saved questions.json")
