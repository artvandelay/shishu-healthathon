"""English reviewer copy of the Hindi mother PDF for call 253.

Writes review/samples/SHISHU-sample-red-flags-en.pdf only.
Does not change SHISHU-sample-red-flags.pdf and does not change mother_report.py.
"""

import html
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGO = ROOT / "code" / "shishu-v2" / "static" / "shishu-logo.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = HERE / "samples" / "SHISHU-sample-red-flags-en.pdf"

NAVY = "#1E2947"
GOLD = "#DAA550"
GREY = "#565F75"
LINE = "#E0E2E5"

CSS = f"""
@page {{
  size: A4;
  margin: 14mm 16mm 16mm;
  @bottom-left {{
    content: "SHISHU";
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 8pt;
    color: #8E95A3;
  }}
  @bottom-right {{
    content: counter(page) " / " counter(pages);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 8pt;
    color: #8E95A3;
  }}
}}
* {{ box-sizing: border-box; }}
html, body {{
  margin: 0;
  padding: 0;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: {NAVY};
  font-size: 11.5pt;
  line-height: 1.4;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}
.top {{ display: flex; align-items: center; gap: 8px; }}
.top img {{ width: 26px; height: 26px; }}
.brand {{
  font-size: 12pt;
  font-weight: 700;
  letter-spacing: 0.14em;
}}
.kind {{ margin-left: auto; font-size: 11pt; font-weight: 700; color: {GOLD}; text-align: right; }}
.kind span {{ display: block; font-weight: 400; font-size: 8.5pt; color: {GREY}; margin-top: 1px; }}
.rule {{ height: 2px; background: {GOLD}; margin: 6px 0 8px; }}
.who {{ margin: 0; font-size: 12pt; font-weight: 700; }}
.meta {{ margin: 1px 0 0; font-size: 10pt; color: {GREY}; }}
.notice {{
  margin: 12px 0 0;
  padding: 8px 12px;
  border: 1.5px solid {NAVY};
  border-left: 5px solid {GOLD};
  font-size: 12pt;
  font-weight: 700;
}}
.notice span {{ display: block; font-weight: 400; font-size: 10.5pt; color: {GREY}; margin-top: 2px; }}
.band {{
  margin: 10px 0 0;
  padding: 8px 12px;
  background: #FBF3E4;
  border-left: 5px solid {GOLD};
}}
h1 {{ margin: 16px 0 8px; font-size: 14pt; }}
ol {{ margin: 0; padding-left: 1.2em; }}
ol li {{ margin: 0 0 7px; }}
ol li::marker {{ color: {GOLD}; font-weight: 700; }}
.note {{ margin: 4px 0 0; font-size: 10.5pt; color: {GREY}; }}
h2 {{ margin: 14px 0 6px; font-size: 13pt; }}
.yes {{ margin: 0; padding: 0; list-style: none; }}
.yes li {{ margin: 0 0 6px; padding-left: 12px; border-left: 2px solid {GOLD}; }}
.page2 {{ break-before: page; }}
.page2 h1 {{ margin-top: 0; }}
.group {{ margin: 12px 0 2px; font-size: 11pt; font-weight: 700; color: {GOLD}; break-after: avoid; }}
.no-row {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 4px 0;
  border-bottom: 1px solid {LINE};
  break-inside: avoid;
}}
.ans {{ flex: 0 0 auto; color: {GREY}; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
.chip {{
  padding: 3px 10px;
  border: 1px solid {LINE};
  border-radius: 99px;
  font-size: 10pt;
  color: {GREY};
}}
"""

QUESTIONS = [
    "Should my child have a thalassemia test?",
    "Does my child need any special test or check-up?",
    "Are my child's weight and height right for this age?",
    "Have all the vaccines been given on time?",
    "Do I need any other test during this pregnancy?",
    "A child in our family died at a young age. Should we ask the doctor for special advice, or for a test, about this child's health or later care?",
]

YES = [
    "On the question about how the baby was born, the mother said the birth was by operation because the baby was very small.",
    "On the question about an early birth, the mother said the baby was born at 8 months.",
    "On the question about the hearing and vision check, the mother said the check was done and it was normal.",
    "On the question about a blood transfusion for the child, the mother said the child has already had blood four times.",
    "On the question about the blood count, the mother said the hemoglobin stays low.",
    "On the question about blood pressure in pregnancy, the mother said her blood pressure was high during the pregnancy.",
    "On the question about a transfusion in pregnancy, the mother said she was given blood one or two weeks before the delivery.",
    "On the question about being related before marriage, the mother said she and her husband were related.",
    "On the question about a pregnancy that ended, the mother said her three earlier pregnancies ended on their own.",
    "On the question about repeated transfusions in the family, the mother said one relative needed blood again and again.",
]

NO_GROUPS = [
    (
        "Birth",
        [
            "Did the baby have to stay in an incubator (glass box) or in the children's ICU?",
            "Was there a large swelling, a boil, or a lump in the armpit at the BCG vaccine site?",
            "Was the lip or the palate split (cleft)?",
        ],
    ),
    (
        "Body and growth",
        [
            "Were there fits, or did sitting, walking, or talking come later than in other children?",
            "Can the child no longer do something the child had already learned?",
            "Is the tummy slowly getting bigger?",
            "After a fall, does a knee or a joint swell and then settle on its own in a few days?",
            "Does a small injury bleed a lot, or do bruises come easily?",
            "Do the private parts not look as they should?",
            "Have there been swellings or lumps on the body since birth?",
            "Are the arms or the legs very short for the body?",
            "Is a thumb or a finger small, or missing?",
            "Is there pain in the arms or the legs again and again, enough to need a hospital?",
        ],
    ),
    (
        "Illness",
        [
            "Has the child been admitted to hospital again and again?",
            "Did an infection need a drip or a medicine in the vein again and again?",
            "After birth, did the child suddenly become very ill and go to emergency?",
            "When a meal is missed, does the child become very sleepy, vomit, or have fits?",
        ],
    ),
    (
        "Pregnancy",
        [
            "Did both the mother and the father have a thalassemia test?",
            "Was the pregnancy scan abnormal?",
            "Did a scan or the birth show a problem in the heart?",
        ],
    ),
    (
        "Family",
        [
            "Did any child in the family die before the age of one year?",
            "Does any child or adult in the family have fits, learn late, or lose a learned skill?",
            "Does any child or adult in the family have blindness from a young age?",
            "Does any child or adult in the family hear poorly from a young age?",
            "Does anyone in the family bleed too much?",
            "Does anyone in the family have cancer?",
            "Does anyone in the family have heart disease?",
            "Does anyone in the family have kidney disease?",
            "Does anyone in the family under about 40 to 50 years have high blood pressure, diabetes, or a stroke?",
            "On the mother's side of the family, does anyone have heavy bleeding, repeated infections, or dry itchy patches on the skin?",
            "Does anyone in the family have swellings or lumps on the body?",
            "Does anyone in the family have the same problem as this child?",
        ],
    ),
]

UNKNOWN = [
    "Did a lot of blood come when the umbilical cord fell?",
    "Did the cord take more than three weeks to fall?",
    "Do the lips or the fingers turn blue?",
    "Does the urine or the stool smell strange?",
]


def _rows() -> str:
    parts: list[str] = []
    for title, items in NO_GROUPS:
        parts.append(f'<div class="group">{html.escape(title)}</div>')
        for question in items:
            parts.append(
                '<div class="no-row"><span>'
                + html.escape(question)
                + '</span><span class="ans">No</span></div>'
            )
    parts.append('<div class="group">No clear answer</div>')
    for question in UNKNOWN:
        parts.append(
            '<div class="no-row"><span>'
            + html.escape(question)
            + '</span><span class="ans">Don\'t know</span></div>'
        )
    parts.append('<div class="group">Not asked yet</div>')
    parts.append('<div class="chips"><span class="chip">Body and growth</span></div>')
    return "".join(parts)


def render_html() -> str:
    questions = "\n".join(f"<li>{html.escape(q)}</li>" for q in QUESTIONS)
    yes = '<ul class="yes">' + "".join(f"<li>{html.escape(line)}</li>" for line in YES) + "</ul>"
    logo = LOGO.as_uri()
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<header>
  <div class="top">
    <img src="{logo}" alt="">
    <div class="brand">SHISHU</div>
    <div class="kind">Call report<span>English copy for reviewers</span></div>
  </div>
  <div class="rule"></div>
  <p class="who">Kriya · 2 years</p>
  <p class="meta">7 November 2026 · 3:44 pm · 13 min 15 sec · Incomplete call</p>
</header>
<div class="notice">This is not a diagnosis.<span>Only a doctor can examine the child and give the correct advice.</span></div>
<div class="band"><b>This call was not finished.</b> Below is only what this call could ask. When there is time, call again and complete the remaining questions.</div>
<h1>Ask the doctor or the health worker (ASHA/ANM) these questions</h1>
<ol>{questions}</ol>
<p class="note">If something here is wrong, or you want to add something, you can call again.</p>
<h2>What was yes on this call</h2>
{yes}
<section class="page2">
  <h1>Detail</h1>
  <p class="meta">Kriya · 2 years</p>
  {_rows()}
</section>
</body>
</html>
"""


def main() -> None:
    document = render_html()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html_path = OUT.with_suffix(".html")
    html_path.write_text(document, encoding="utf-8")
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={OUT}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    html_path.unlink()
    hindi = OUT.with_name("SHISHU-sample-red-flags.pdf")
    if not hindi.is_file():
        raise SystemExit(f"Hindi PDF missing: {hindi}")
    print(f"{OUT.name} {OUT.stat().st_size} bytes")
    print(f"kept {hindi.name} {hindi.stat().st_size} bytes")


if __name__ == "__main__":
    main()
