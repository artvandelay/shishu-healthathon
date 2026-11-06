import os, time
from dotenv import load_dotenv
from google import genai
load_dotenv(override=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
prompt = 'You are a clinic intake assistant. The mother says her baby has had a cough for 3 days. Reply ONLY with JSON: {"reply": "<one short question>", "done": false}'
results = []
for m in client.models.list():
    name = m.name.replace("models/", "")
    if "generateContent" not in (m.supported_actions or []):
        continue
    if any(x in name for x in ("audio", "live", "tts", "image", "embedding", "veo", "robotics")):
        continue
    t = time.time()
    try:
        client.models.generate_content(model=name, contents=prompt)
        results.append((time.time() - t, name))
        print(f"{time.time()-t:5.1f}s  {name}")
    except Exception as e:
        print("FAILS", name, str(e)[:50])
print("\nFastest first:")
for s, n in sorted(results):
    print(f"{s:5.1f}s  {n}")
