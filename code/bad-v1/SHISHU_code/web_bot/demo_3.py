import os, json, sqlite3, time, random, re, hashlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)
app = FastAPI()
gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CLINIC = os.environ.get("CLINIC_NAME", "the clinic")
MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview"]
# Model used ONLY for "what is <disease>?" answers. It must support Google Search grounding (Gemini does, Gemma does not).
INFO_MODEL = os.environ.get("INFO_MODEL", "gemini-2.5-flash")
QUESTIONS = json.load(open("questions.json", encoding="utf-8"))
QTEXT = {q.get("id"): q.get("question") for q in QUESTIONS}

# ---------- DB ----------
db = sqlite3.connect("demo.db", check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS appointments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mother_name TEXT, doctor_name TEXT, appt_time TEXT, status TEXT DEFAULT 'intake',
  answers TEXT DEFAULT '{}', summary_doctor TEXT, summary_mother TEXT, alerts TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, appt_id INTEGER, role TEXT, content TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP);
""")

try:  # older demo.db files do not have the profile column yet
    db.execute("ALTER TABLE appointments ADD COLUMN profile TEXT DEFAULT '{}'")
    db.commit()
except sqlite3.OperationalError:
    pass

def get_appt(appt_id):
    a = db.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
    if not a:
        raise HTTPException(404, "Appointment not found")
    return a

def save_msg(appt_id, role, content):
    db.execute("INSERT INTO messages(appt_id,role,content) VALUES(?,?,?)", (appt_id, role, content))
    db.commit()

def merged_messages(appt_id, limit=40):
    rows = db.execute("SELECT role,content FROM messages WHERE appt_id=? ORDER BY id DESC LIMIT ?",
                      (appt_id, limit)).fetchall()
    out = []
    for r in reversed(rows):
        m = {"role": r["role"], "content": r["content"]}
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n" + m["content"]
        else:
            out.append(m)
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(appointment booked)"})
    return out

# ---------- LLM ----------
def call_json(system, messages):
    convo = "\n".join(f"{'MOTHER' if m['role']=='user' else 'ASSISTANT'}: {m['content']}" for m in messages)
    prompt = f"{system}\n\n--- CONVERSATION SO FAR ---\n{convo}\n\nNow respond with ONLY the JSON object."
    last_err = None
    for attempt in range(6):
        model = MODELS[0] if attempt < 3 else MODELS[1]
        try:
            r = gem.models.generate_content(model=model, contents=prompt)
            t = r.text
            return json.loads(t[t.index("{"): t.rindex("}") + 1])
        except Exception as e:
            last_err = e
            print(f"LLM error ({model}):", str(e)[:150])
            time.sleep(2 * (attempt + 1) + random.random())
    raise last_err

# ---------- Question relevance ----------
TAG_PROMPT = """Tag each question with who it applies to. Use exactly one tag per question:
- "child": asks about her child/children/baby (growth, feeding, vaccines, child's symptoms, school, birth history of a child)
- "marriage": asks about husband/spouse, pregnancy, delivery, family planning, or family relationships such as parents being related
- "all": anything else (general symptoms, medicines, family history, the patient herself)
Respond with ONLY a JSON object mapping each question id to its tag. Questions: """

def _fallback_tag(text):
    t = (text or "").lower()
    if re.search(r"\b(child|children|baby|babies|infant|newborn|toddler|kid|kids|son|daughter|breastfe\w*|vaccin\w*|school)\b", t):
        return "child"
    if re.search(r"\b(pregnan\w*|husband|spouse|marriage|married|consanguin\w*|miscarriage|delivery|antenatal|family planning)\b", t):
        return "marriage"
    return "all"

def load_tags():
    sig = hashlib.md5(json.dumps(QUESTIONS, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    try:
        c = json.load(open("question_tags.json", encoding="utf-8"))
        if c.get("sig") == sig:
            return c["tags"]
    except Exception:
        pass
    tags, ok = {}, False
    try:
        items = [{"id": q.get("id"), "question": q.get("question")} for q in QUESTIONS]
        out = call_json(TAG_PROMPT + json.dumps(items, ensure_ascii=False),
                        [{"role": "user", "content": "Tag the questions."}])
        tags = {str(k): v for k, v in out.items() if v in ("all", "marriage", "child")}
        ok = True
    except Exception as e:
        print("tagging failed, using keyword fallback:", str(e)[:150])
    for q in QUESTIONS:
        tags.setdefault(str(q.get("id")), _fallback_tag(q.get("question")))
    if ok:  # cached so tagging runs once; edit question_tags.json by hand to fix any tag, delete it to re-tag
        json.dump({"sig": sig, "tags": tags}, open("question_tags.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return tags

TAGS = load_tags()

def relevant_questions(profile, answers):
    allow = {"all"}
    if profile.get("marital") in ("married", "about_to_be_married", "previously_married"):
        allow.add("marriage")
    if profile.get("has_kids") in ("yes", "expecting"):
        allow.add("child")
    return [q for q in QUESTIONS
            if TAGS.get(str(q.get("id")), "all") in allow and str(q.get("id")) not in answers]

# ---------- Disease information from reputable sources ----------
INFO_PROMPT = """Explain "{topic}" to a mother with no medical background, in simple {language}.
Use only facts from reputable health sources (WHO, NHS, NIH/MedlinePlus, CDC, Orphanet, Indian Ministry of Health / NHM / ICMR).
Write 4 to 6 short sentences: what it is, how it is usually caused or inherited, common signs, and that a doctor can guide testing and care.
Do NOT say whether she or her child has it. Do NOT mention medicines, doses or treatment choices. Plain text, no markdown."""

def explain(topic, language):
    try:
        r = gem.models.generate_content(
            model=INFO_MODEL,
            contents=INFO_PROMPT.format(topic=topic, language=language or "English"),
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]))
        text = (r.text or "").strip()
        if not text:
            return None
        names = []
        gm = r.candidates[0].grounding_metadata if r.candidates else None
        for ch in (getattr(gm, "grounding_chunks", None) or []):
            t = ch.web.title if ch.web else None
            if t and t not in names:
                names.append(t)
        return text + (f"\n\n(Sources: {', '.join(names[:3])})" if names else "")
    except Exception as e:
        print("info error:", str(e)[:150])
        return None

# ---------- Prompts ----------
INTAKE_PROMPT = """You are SHISHU, a warm, patient pre-appointment intake assistant for {clinic}, chatting with a mother/guardian before their appointment with {doctor}.

RULES
- You collect information only. NEVER diagnose, say whether she or her child has a condition, or discuss medicines, doses or treatment. If she asks for those, kindly say the doctor will answer during the visit, then continue.
- Ask ONE or two short questions at a time, in simple language.
- LANGUAGE: look at the language AND script of her LATEST message and reply in exactly that. Examples: Kannada script, Kannada written in English letters, Hindi, Hinglish, Tamil, Telugu, Malayalam, Marathi, Bengali, English. If she writes in English letters, reply in English letters. Set "language" in profile (e.g. "Kannada", "Hindi", "English"). Keep answers she gives in her own words.
- GENERAL INFORMATION: if she asks a general question about a disease or condition (for example "what is thalassemia?"), set "info_topic" to the short topic name. Do NOT explain it yourself; a verified explanation from reputable sources is added automatically before your reply. Your "reply" must then be only a short friendly bridge plus your next question. This is only for "what is / why does it happen / what are the signs" questions. Leave info_topic "" for anything else, including "does my child have X?" and medicine questions.
- If a question was only half answered, ask the unanswered part again.
- If an answer is vague, ask one brief follow-up (for example "since when?" or "how many times a day?").
- If she mentions any emergency sign (difficulty breathing, blue lips, seizure, unresponsive child, very high fever in a young infant, severe bleeding), set "emergency": true and tell her to go to the nearest hospital or call emergency services immediately.

STEP 1 - ONBOARDING (always first, in this order; skip anything already in PROFILE):
 a. Name. Ask: "May I know your name?" -> profile.name
 b. Ask: "Are you married or about to be married?" -> profile.marital = married | about_to_be_married | not_married | previously_married
 c. ONLY if marital is married or previously_married, ask: "Do you have kids?" -> profile.has_kids = yes | no | expecting
 d. Ask: "What is the main problem you'd like the doctor to look at?" -> profile.concern (short, in her words). If she only wants information (for example about a rare disease), set concern to that.
Ask one onboarding question per message. If she answers several at once, record all of them and ask the next missing one. Translate these questions into her language.

STEP 2 - QUESTION BANK (only after a to d are in PROFILE):
- PENDING QUESTIONS below are already filtered for her situation. Ask ONLY the ones relevant to her main concern and silently skip the rest.
- Never ask about children, babies, pregnancy or a spouse unless the question is in PENDING QUESTIONS.
- If she only wants general information, ask few or none, then ask if there is anything else she wants the doctor to know.
- When nothing relevant is left (or she says she has nothing more), thank her, say a summary will be shared with her and the doctor, and set "done": true.

PROFILE: {profile}
PENDING QUESTIONS: {questions}
ANSWERS SO FAR: {answers}

Respond with ONLY this JSON, nothing else:
{{"reply": "<message to send>", "profile": {{"name": "", "marital": "", "has_kids": "", "concern": "", "language": ""}}, "answers": {{"<question id>": "<answer in her words>"}}, "info_topic": "", "done": false, "emergency": false}}
In "profile" and "answers" include only NEW or UPDATED values; leave out empty ones."""

SUMMARY_PROMPT = """Using the profile and the answers below, write two summaries. Do not diagnose or speculate.
1. "doctor": in English. Concise clinical-style summary: who she is (name, marital status, kids if known), chief complaint or what she wants to know, onset and duration, symptoms, relevant history, medications, anything unanswered, and any red flags.
2. "mother": in the mother's language (given below; translate the fixed closing sentences too). A simple friendly recap of what she told us, if there is red flag then ending with" It would be a good idea to have some of the things you mentioned checked by a health worker. This might not be serious, but it is better to be sure. Please show this message to your ASHA/ANM, or visit the nearest health centre."  else " Based on what you have shared, there is no need to go anywhere right now. If you notice any of the things we asked about in the future, please come back and let us know, or speak to your health worker."
Respond with ONLY this JSON: {"doctor": "...", "mother": "..."}"""

FOLLOWUP_PROMPT = """You are the follow-up assistant for {clinic}. A mother ({mother}) had an appointment with {doctor}.
Earlier intake summary:
{summary}

Check in kindly: ask how the child is doing, whether the doctor's advice is being followed, and whether symptoms improved or worsened. Do not repeat questions already answered. Never diagnose or change medicines.
If symptoms worsen or red flags appear, set "escalate": true and advise contacting the clinic or urgent care.
Reply in the mother's language. Keep messages short.
Respond with ONLY this JSON: {{"reply": "...", "escalate": false, "note": "<one-line note for the doctor, or empty>"}}"""

def add_alert(appt_id, text):
    a = get_appt(appt_id)
    db.execute("UPDATE appointments SET alerts=? WHERE id=?", ((a["alerts"] + "\n" + text).strip(), appt_id))
    db.commit()

def finalize(appt_id):
    a = get_appt(appt_id)
    profile = json.loads(a["profile"] or "{}")
    answered = {QTEXT.get(k, k): v for k, v in json.loads(a["answers"]).items()}
    s = call_json(SUMMARY_PROMPT + f"\n\nMOTHER'S LANGUAGE: {profile.get('language', 'English')}"
                  f"\nPROFILE: {json.dumps(profile, ensure_ascii=False)}"
                  f"\nANSWERS (question: answer): {json.dumps(answered, ensure_ascii=False)}\nALERTS: {a['alerts']}",
                  [{"role": "user", "content": "Write the summaries."}])
    db.execute("UPDATE appointments SET status='completed', summary_doctor=?, summary_mother=? WHERE id=?",
               (s["doctor"], s["mother"], appt_id))
    db.commit()

# ---------- API ----------
class Book(BaseModel):
    mother_name: str = ""
    doctor_name: str
    appt_time: str

class Chat(BaseModel):
    appt_id: int
    message: str

@app.post("/api/book")
def book(b: Book):
    cur = db.execute("INSERT INTO appointments(mother_name,doctor_name,appt_time) VALUES(?,?,?)",
                     (b.mother_name, b.doctor_name, b.appt_time))
    db.commit()
    welcome = (f"Hello! This is SHISHU. Your appointment with {b.doctor_name} is on "
               f"{b.appt_time}. To help the doctor prepare, I'll ask a few quick questions. I can't give medical "
               f"advice, I only collect information for the doctor.\n\nMay I know your name?")
    save_msg(cur.lastrowid, "assistant", welcome)
    return {"appt_id": cur.lastrowid}

@app.get("/api/appointments")
def list_appts():
    rows = db.execute("SELECT id, mother_name, doctor_name, appt_time, status FROM appointments ORDER BY id DESC LIMIT 30").fetchall()
    return [dict(r) for r in rows]

@app.get("/api/appointments/{appt_id}")
def view(appt_id: int):
    a = get_appt(appt_id)
    msgs = db.execute("SELECT role, content FROM messages WHERE appt_id=? ORDER BY id", (appt_id,)).fetchall()
    answers = json.loads(a["answers"])
    profile = json.loads(a["profile"] or "{}")
    labels = {"name": "Name", "marital": "Marital status", "has_kids": "Kids", "concern": "Main concern", "language": "Language"}
    prof_rows = [{"q": labels[k], "a": profile[k]} for k in labels if profile.get(k)]
    return {"id": a["id"], "mother_name": a["mother_name"], "doctor_name": a["doctor_name"],
            "appt_time": a["appt_time"], "status": a["status"],
            "answers": prof_rows + [{"q": QTEXT.get(k, k), "a": v} for k, v in answers.items()],
            "alerts": a["alerts"], "summary_doctor": a["summary_doctor"], "summary_mother": a["summary_mother"],
            "messages": [dict(m) for m in msgs]}

@app.post("/api/chat")
def chat(c: Chat):
    a = get_appt(c.appt_id)
    if a["status"] == "completed":
        reply = "Your intake is complete. The doctor will review the summary before your visit."
        save_msg(c.appt_id, "user", c.message); save_msg(c.appt_id, "assistant", reply)
        return {"ok": True}
    save_msg(c.appt_id, "user", c.message)
    try:
        if a["status"] == "intake":
            profile = json.loads(a["profile"] or "{}")
            answers = json.loads(a["answers"])
            system = INTAKE_PROMPT.format(clinic=CLINIC, doctor=a["doctor_name"],
                                          profile=json.dumps(profile, ensure_ascii=False),
                                          questions=json.dumps(relevant_questions(profile, answers), ensure_ascii=False),
                                          answers=json.dumps(answers, ensure_ascii=False))
            out = call_json(system, merged_messages(c.appt_id))
            for k, v in (out.get("profile") or {}).items():
                if k in ("name", "marital", "has_kids", "concern", "language") and v:
                    profile[k] = v
            if profile.get("marital") in ("not_married", "about_to_be_married"):
                profile["has_kids"] = "no"          # kids question is only asked to married people
            answers.update(out.get("answers") or {})
            db.execute("UPDATE appointments SET answers=?, profile=? WHERE id=?",
                       (json.dumps(answers, ensure_ascii=False), json.dumps(profile, ensure_ascii=False), c.appt_id))
            if profile.get("name"):
                db.execute("UPDATE appointments SET mother_name=? WHERE id=?", (profile["name"], c.appt_id))
            db.commit()
            reply = out["reply"]
            topic = (out.get("info_topic") or "").strip()
            if topic:
                info = explain(topic, profile.get("language"))
                if info:
                    reply = info + "\n\n" + reply
                else:
                    reply = ("I could not fetch reliable information on this right now. The doctor will be happy to "
                             "explain it during your visit.\n\n") + reply
            save_msg(c.appt_id, "assistant", reply)
            if out.get("emergency"):
                add_alert(c.appt_id, f"URGENT: possible emergency signs reported: {c.message}")
            if out.get("done"):
                finalize(c.appt_id)
        else:  # followup
            system = FOLLOWUP_PROMPT.format(clinic=CLINIC, mother=a["mother_name"], doctor=a["doctor_name"],
                                            summary=a["summary_doctor"])
            out = call_json(system, merged_messages(c.appt_id))
            save_msg(c.appt_id, "assistant", out["reply"])
            if out.get("escalate") or out.get("note"):
                add_alert(c.appt_id, f"{'URGENT: ' if out.get('escalate') else ''}Follow-up: {out.get('note') or c.message}")
    except Exception as e:
        print("error:", e)
        save_msg(c.appt_id, "assistant", "Sorry, something went wrong on our side. Please try again in a moment.")
    return {"ok": True}

@app.post("/api/followup/{appt_id}")
def followup(appt_id: int):
    a = get_appt(appt_id)
    if not a["summary_doctor"]:
        raise HTTPException(400, "Finish the intake first")
    db.execute("UPDATE appointments SET status='followup' WHERE id=?", (appt_id,)); db.commit()
    save_msg(appt_id, "assistant", f"Hello {a['mother_name']}, this is {CLINIC}. How is your child doing since the visit "
                                  f"with {a['doctor_name']}? Is there any change in the symptoms?")
    return {"ok": True}

# ---------- Web page ----------
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Pre-visit assistant demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box} body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f4f6;color:#1f2937}
header{background:#075e54;color:#fff;padding:10px 16px;font-weight:600}
.bar{display:flex;gap:8px;flex-wrap:wrap;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;align-items:center}
.bar input,.bar select{padding:7px;border:1px solid #ccc;border-radius:6px} button{padding:8px 12px;border:0;border-radius:6px;background:#128c7e;color:#fff;cursor:pointer}
button.alt{background:#4b5563} button:disabled{opacity:.5;cursor:not-allowed}
.main{display:flex;gap:14px;padding:14px 16px;align-items:flex-start;flex-wrap:wrap}
.phone{flex:1 1 380px;max-width:520px;background:#fff;border-radius:12px;box-shadow:0 1px 4px #0003;overflow:hidden}
.phone h3{margin:0;padding:10px 14px;background:#128c7e;color:#fff;font-size:15px}
#chat{height:430px;overflow-y:auto;padding:12px;background:#ece5dd}
.msg{max-width:80%;padding:8px 11px;margin:6px 0;border-radius:10px;white-space:pre-wrap;line-height:1.35;font-size:14px}
.assistant{background:#fff;margin-right:auto} .user{background:#dcf8c6;margin-left:auto}
.send{display:flex;gap:6px;padding:8px;background:#f0f0f0} .send input{flex:1;padding:9px;border:1px solid #ccc;border-radius:18px}
.doctor{flex:1 1 380px;background:#fff;border-radius:12px;box-shadow:0 1px 4px #0003;padding:14px}
.doctor h3{margin:0 0 8px} .doctor h4{margin:14px 0 6px;color:#075e54}
.alert{background:#fee2e2;border-left:4px solid #dc2626;padding:8px;border-radius:4px;white-space:pre-wrap}
pre{white-space:pre-wrap;background:#f9fafb;padding:10px;border-radius:6px;font-family:inherit;margin:0}
.qa{font-size:13px;margin:4px 0} .qa b{color:#374151} .muted{color:#6b7280;font-size:13px}
</style></head><body>
<header>Pre-visit assistant (demo) - collects information only, does not diagnose</header>
<div class="bar">
  <input id="mn" placeholder="Mother's name (optional)" value="">
  <input id="dn" placeholder="Doctor's name" value="Dr. Test">
  <input id="at" placeholder="Appointment time" value="Fri 6 Nov, 11:00 AM">
  <button onclick="book()">Book appointment</button>
  <select id="old" onchange="openOld()"><option value="">Open earlier appointment...</option></select>
  <button class="alt" id="fu" onclick="followup()" disabled>Start follow-up</button>
</div>
<div class="main">
  <div class="phone"><h3 id="title">SHISHU</h3><div id="chat"></div>
    <div class="send"><input id="txt" placeholder="Type the mother's reply..." onkeydown="if(event.key==='Enter')send()"><button onclick="send()" id="sendbtn">Send</button></div></div>
  <div class="doctor"><h3>Doctor's view</h3><div class="muted" id="meta">No appointment open.</div>
    <div id="alerts"></div>
    <h4>Answers collected so far</h4><div id="answers" class="muted">-</div>
    <h4>Summary for doctor</h4><pre id="sd">Available when the intake finishes.</pre>
    <h4>Summary sent to mother</h4><pre id="sm">Available when the intake finishes.</pre></div>
</div>
<script>
let apptId=null, busy=false;
async function api(path, method, body){
  const r = await fetch(path,{method:method||'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  if(!r.ok){ throw new Error(await r.text()); } return r.json();
}
async function loadList(){
  const list = await api('/api/appointments'); const sel=document.getElementById('old');
  sel.innerHTML='<option value="">Open earlier appointment...</option>';
  list.forEach(a=>{const o=document.createElement('option');o.value=a.id;o.textContent='#'+a.id+' '+(a.mother_name||'New')+' ('+a.status+')';sel.appendChild(o);});
}
async function book(){
  const d = await api('/api/book','POST',{mother_name:mn.value,doctor_name:dn.value,appt_time:at.value});
  apptId=d.appt_id; await refresh(); loadList();
}
async function openOld(){ const v=document.getElementById('old').value; if(v){apptId=parseInt(v); await refresh();} }
async function refresh(){
  if(!apptId) return;
  const a = await api('/api/appointments/'+apptId);
  const chat=document.getElementById('chat'); chat.innerHTML='';
  a.messages.forEach(m=>{const d=document.createElement('div');d.className='msg '+m.role;d.textContent=m.content;chat.appendChild(d);});
  chat.scrollTop=chat.scrollHeight;
  document.getElementById('title').textContent="SHISHU - "+a.mother_name;
  document.getElementById('meta').textContent='Appointment #'+a.id+' with '+a.doctor_name+', '+a.appt_time+' - status: '+a.status;
  const al=document.getElementById('alerts'); al.innerHTML='';
  if(a.alerts){const d=document.createElement('div');d.className='alert';d.textContent=a.alerts;al.appendChild(d);}
  const ans=document.getElementById('answers'); ans.innerHTML='';
  if(!a.answers.length){ans.textContent='-';}
  a.answers.forEach(x=>{const d=document.createElement('div');d.className='qa';const b=document.createElement('b');b.textContent=x.q+' ';d.appendChild(b);d.appendChild(document.createTextNode(x.a));ans.appendChild(d);});
  document.getElementById('sd').textContent=a.summary_doctor||'Available when the intake finishes.';
  document.getElementById('sm').textContent=a.summary_mother||'Available when the intake finishes.';
  document.getElementById('fu').disabled = !(a.status==='completed');
}
async function send(){
  if(!apptId||busy) return; const t=txt.value.trim(); if(!t) return;
  busy=true; sendbtn.disabled=true; txt.value='';
  const chat=document.getElementById('chat');
  const u=document.createElement('div');u.className='msg user';u.textContent=t;chat.appendChild(u);
  const w=document.createElement('div');w.className='msg assistant';chat.appendChild(w);chat.scrollTop=chat.scrollHeight;
  const phrases=['SHISHU is thinking','SHISHU is reading your message','SHISHU is preparing the next question'];
  let tick=0; const show=()=>{w.textContent=phrases[Math.floor(tick/8)%phrases.length]+'.'.repeat(tick%4);};
  show(); const timer=setInterval(()=>{tick++;show();},400);
  try{ await api('/api/chat','POST',{appt_id:apptId,message:t}); }catch(e){ alert('Error: '+e.message); }
  clearInterval(timer);
  await refresh(); busy=false; sendbtn.disabled=false; txt.focus();
}
async function followup(){ await api('/api/followup/'+apptId,'POST'); await refresh(); }
loadList();
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
