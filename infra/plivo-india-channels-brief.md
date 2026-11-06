# SHISHU: what Plivo can do in India, by channel
Prepared 13 Oct 2026 (IST) for a pilot of 30–50 calls over 2 months. Our FastAPI backend does STT, LLM and TTS. Plivo KYC goes under Jigar's company, and we rent a Plivo +91 number.
Tags: **[V]** = checked against a primary or official source today. **[S]** = secondary source. **[U]** = not verified, inference only.

## TL;DR
1. **Which number to rent:** a Plivo **landline-series number (080/022)**. It handles inbound calls, outbound *service* calls and WhatsApp. **We do not need 140 or 160:** 140 is for promotional calls only, and Plivo offers 160 only to BFSI firms (RBI/SEBI/IRDAI/PFRDA). KYC needs the COI and GST certificate, and the automated review usually finishes in about 5 minutes. [V]
2. **SMS through Plivo in India is not possible.** Plivo's India pricing page lists domestic SMS as "Not Supported" for both outbound and inbound. [V] Texting has to go through WhatsApp, or SMS has to come from a separate Indian DLT provider.
3. **WhatsApp through Plivo works.** You get text, media and voice notes, plus WhatsApp *calling* that streams to our WebSocket using the same `<Stream>` XML. Costs are Meta's rate plus ₹0.10 per message and **₹999 a month per WABA** in Plivo fees. [V]
4. **The real regulatory risks:**
   - (a) TRAI's **Third Amendment (18 Sep 2026)** counts AI-voice outbound calls as "A2P". They **must be declared in advance to the operator, with our CLIs listed**, or they are treated as spam (UCC). [V]
   - (b) After an initial service call, follow-up calls about "ongoing use" need **explicit consent, which lasts only 7 days** before she has to renew it. [V, Plivo's summary of TCCCPR]
5. **Pilot cost:** about **₹1,000** for phone only, or about **₹3,000–3,500** with Plivo WhatsApp. Both exclude 18% GST. Money is not the constraint here.

---

## 1. Inbound voice calls to the +91 number
| | |
|---|---|
| **Can do** | Mothers call our number. Plivo hits our Answer URL, and we return `<Stream bidirectional="true" keepCallAlive="true">wss://…</Stream>`. Plivo then streams about 20 ms audio frames (mulaw or L16, 8/16 kHz) plus DTMF events to our WebSocket, and we stream TTS back. [V] Recording, conferences (a counsellor can join, or join muted to listen in), and transfer via `<Dial>` are all supported. Concurrency is a per-account channel setting, not a property of the number. [U, confirm the default with Plivo] |
| **Rules** | India-registered business only. **Media anchoring:** both legs of every call must be in India, or the call fails with `violates_media_anchoring`. That covers conference participants and transfer targets, so a counsellor abroad would be dropped. [V] This only affects call legs; our WebSocket server can sit anywhere [U]. India-region hosting is still advisable. A separate **India data-region Plivo account** is required. [V] |
| **Consent** | Inbound is not commercial communication from us, so TCCCPR consent doesn't apply. DPDP notice plus consent is still needed for recording and health data (see §7). |
| **Stuck points** | KYC rejection if the COI or GST don't match. Our backend has to be on public WSS. Counsellor transfers must go to Indian numbers. |
| **Cost** | ₹0.38/min, billed in 30-second pulses. Number ₹200/month. Streaming, recording, conferencing and AMD all ₹0. Recording storage is free for 90 days, then $0.0004/min/month. Optional: noise cancellation ₹0.12/min. Skip Plivo's ASR (₹1.70 per 15 s) and transcription (₹0.81/min). Source: plivo.com/voice/pricing/in [V] |

## 2. Outbound voice calls from the +91 number
| | |
|---|---|
| **Can do** | Calls placed by the API from our rented landline-series number, with the same streaming, recording and conferencing as inbound. The caller ID must be a Plivo-rented Indian number. [V] |
| **Rules: number series** | **Landline series (080/022): service and transactional calls only, no promotional content.** 140: promotional only, needs Tata DLT PE+TM registration, the TM must be registered in Mumbai or Karnataka, and setup takes 5–10 business days. 160: BFSI only. "Using the wrong series is itself a regulatory violation." [V] TRAI's Feb 2025 amendment moves service and transactional calls to the **1600 series**. TRAI's Nov 2025 Direction makes that mandatory only for RBI, SEBI and PFRDA entities so far. [V] Health and education entities are not yet mandated [U, watch for future TRAI directions]. |
| **Rules: TCCCPR definitions** (Plivo's summary of the 2026 Third Amendment) [V] | **Service call:** about a product or service the customer already has. No explicit consent needed. A call that helps her keep *using* the service counts as a service call **only with explicit consent, valid for 7 days or until she revokes it, renewable 7 days at a time**. **Transactional call:** within 30 minutes of a transaction she started. **Inferred consent via relationship:** a written or digital inquiry from her in the last 7 days, or an application in the last 3 months. **Explicit consent** must be verifiable and registered with a Consent Registrar or in the Consent Register. **Cold calling is prohibited.** |
| **Rules: A2P / AI voice** | Third Amendment (TRAI PR 119/2026, 18 Sep 2026): A2P calls include "pre-recorded/artificial voice". Every entity "must pre-declare such use to its TSP along with the details of CLIs… A2P calls made without the required prior declaration will be treated as UCC." There is a termination charge of up to ₹0.05/min on A2P calls from non-designated series. [V] **How Plivo collects this declaration is unknown [U]. Ask Plivo support in writing.** |
| **Rules: NCPR/DND, calling hours** | DND or customer preferences block *promotional* communication. Service calls with valid consent go through. Default blocked time bands for promotional calls are 21:00–10:00. **Service and transactional calls are not time-band restricted** [S, telleo.ai; TRAI consultation]. Good practice is to call 10:00–19:00 at a time she chose. Opt-out means we cannot ask for consent again for 90 days (Feb 2025 amendment). [V] |
| **Enforcement** | 5 or more complaints in 10 days triggers action. Under the 2026 amendment, 3 complaints plus an AI flag is enough. First violation bars outgoing calls for 15 days. Repeat violations mean disconnection across all operators for 1 year plus blacklisting. [V] |
| **Consent** | We do not cold-call. Either (a) she calls first or registers digitally (for example a form or a WhatsApp message), which gives inferred consent for 7 days, or (b) we record explicit consent ("can we call you back on this number for follow-ups?"), which is valid 7 days and renewed on each call. Keep timestamped audio and logs as proof. Plivo's UCC page says consent proof is demanded when a number is suspended. [V] |
| **Stuck points** | (1) The A2P declaration process [U]. (2) The 7-day explicit-consent window for periodic callbacks, which has to be renewed on every call. (3) Whether a free health-education pilot counts as "commercial communication" at all [U, legal opinion]. Treat it as if it does. (4) Calls from unknown 080 numbers get low pickup rates and spam-app tagging. Pre-schedule calls by WhatsApp. |
| **Cost** | ₹0.38/min, 30-second pulse. No DLT fee for landline-series service calls [U, Plivo doesn't list one]. Tata DLT fees for 140/160 are irrelevant to us. |

## 3. SMS
| | |
|---|---|
| **Can do** | **Nothing domestic through Plivo.** plivo.com/sms/pricing/in: Domestic outbound "Not Supported", inbound "Not Supported". Only international (ILDO) at $0.08/SMS, which doesn't apply to us. [V] |
| **Rules (if done elsewhere)** | DLT on any operator portal (Jio, Vilpower/Vi, Airtel, Tata): register as a Principal Entity, a 6-character header (sender ID), content templates (Transactional / Service-Implicit / Service-Explicit / Promotional) and consent templates. Service-Explicit needs registered consent. Indian DLT messaging is one-way, and replies to headers aren't possible. [S, general knowledge] |
| **Consent** | Service-Implicit for messages about the service she has started (like a summary link). Service-Explicit or Promotional for anything else. |
| **Stuck points** | Needs a second vendor (for example MSG91, Gupshup or Exotel) plus DLT. Template approval typically takes 1–3 days, and PE registration about ₹5,900 one-time on most portals (Jio is reportedly free) [U, not checked today]. **Recommendation: skip SMS for the pilot.** |
| **Cost** | Indian DLT SMS is typically ₹0.12–0.25/SMS plus DLT scrubbing charges [U]. |

## 4. WhatsApp messaging through Plivo (Plivo is a Meta BSP)
| | |
|---|---|
| **Can do** | Two-way text, images, documents, **voice notes in and out** (media messages, no extra Plivo charge [V]) and interactive buttons, all via webhook to our backend. Inbound voice notes arrive as media for our own STT. |
| **Number** | Plivo's WhatsApp-calling docs use an 080 Plivo number (`918035737458`) as the example WhatsApp number. That suggests **the same rented landline number can serve both voice and WhatsApp** [U, confirm]. The number must not be registered on any other WhatsApp app, or must be deleted from it first. [S] |
| **Rules** | Meta Business verification and display-name approval are needed to scale. Unverified accounts usually start with limited sending (about 250 business-initiated conversations a day), which is fine for us [S]. **24-hour customer-service window:** opened when she messages *or calls* us, or accepts our call. Inside it we can send free-form messages. Outside it, only approved templates. [V Plivo FAQ] Template categories: Marketing / Utility / Authentication. "Service" means free-form replies inside the window. Our summary and check-in templates should be Utility. Meta can re-categorize anything that looks promotional as Marketing. |
| **Health policy** | WhatsApp Business Policy (whatsappbusiness.com/policy): prohibits *promoting or selling* medical and healthcare products and drugs. Also says "Don't use WhatsApp for telemedicine or to send or request any health related information, **if applicable regulations prohibit** distribution… to systems that do not meet heightened requirements." Prohibits soliciting sensitive identifiers (ID card numbers and similar), and prohibits discrimination based on genetic condition. [V/S via search excerpt] India has no HIPAA-style ban, so health education is allowed [U, legal read]. **Keep WhatsApp content general:** a summary of the education and next steps, the counsellor's contact. Don't put detailed family genetic history in chat, and don't ask for Aadhaar. |
| **Consent** | Opt-in is required before a business-initiated message: she gives her number and agrees on the recorded call to receive messages from "SHISHU". Honour STOP. |
| **Stuck points** | Business verification can take days to weeks. Templates must be approved, usually minutes to 48 hours. Display name has to match the brand. ₹999/month WABA fee. Since **1 Oct 2026, Utility templates sent inside the window are reportedly charged** [S, pelocal/zeebiz]. |
| **Cost (India, plivo.com/whatsapp/pricing/in [V], matches Meta's 1 Oct 2026 INR card [S])** | Marketing ₹0.8631, Utility ₹0.1150, Authentication ₹0.1150, Service ₹0.1150 after **1,000 free service messages a month**. **Plus Plivo fees: ₹0.10 per message (inbound and outbound) and ₹999 a month per WABA.** Exclusive of GST. |

## 5. WhatsApp calling through Plivo
| | |
|---|---|
| **Can do** | Inbound calls when she taps the call button in our chat. Outbound (business-initiated) calls to users who have given permission. Same Plivo Call API and XML: the Answer URL returns `<Stream>` → **streams to our WebSocket**. Outbound uses `<Dial callType="whatsapp" callerId="91…">`. Turn it on in the console with "Enable WhatsApp Calling". [V plivo.com/docs/messaging/concepts/whatsapp/whatsapp-calling] India is supported for both directions. Outbound is blocked only in the US, Canada, Turkey, Egypt, Nigeria and Vietnam. [V] |
| **Permission rules** (Meta Cloud API, from a docs mirror [S]; developers.facebook.com returned 403 to our fetch) | Request permission with a free-form interactive message (inside the window) or a template. **Max 1 request per 24 h and 2 per 7 days**; limits reset after any connected call. Temporary permission lasts **7 days (168 h) from approval**. *Permanent* permission is possible if she chooses it. A request expires 7 days after she answers or after delivery. **A user-initiated call automatically grants temporary permission** if callback is enabled (`response_source: automatic`). **Max 10 connected business-initiated calls per 24 h** per user. 2 unanswered calls in a row triggers a prompt to reconsider; 4 in a row **revokes permission**. |
| **Consent** | The WhatsApp permission tap *is* the platform consent. TCCCPR does not cover OTT calls [U]. DPDP still applies. |
| **Stuck points** | Requires a verified WABA linked to Plivo. Mother needs a smartphone with data. Call quality depends on her data connection. |
| **Cost** | ₹0.45/min inbound, ₹0.75/min outbound, streaming free, recording and conferencing ₹0. Recording storage free for 90 days. [V plivo.com/whatsapp-call/pricing/in] |

## 6. Number and account basics [V]
- KYC needs only the **COI plus GST certificate**. Choose **"Direct Brand"** as the business type.
- Use a separate India-region Plivo account.
- The application can be suspended if UCC complaints aren't resolved.

## 7. DPDP Act 2023 / Rules 2025 and recording norms
- The **DPDP Rules were notified 13 Nov 2025**. The substantive obligations (notice in Rule 3, children's data in Rule 10, security and breach rules) **take effect 13 May 2027**. [S, gazette/PIB via search] Our pilot runs before that, but **build to the standard anyway**: ethics committees and the jury will expect it, and TCCCPR now says consent "does not exempt you from DPDP". [V]
- **Notice + consent:** a clear, separate, plain-language notice listing what data, for what purpose, how to withdraw, and how to complain to the Board. Consent must be free, specific and unambiguous, limited to that purpose, and erased when the purpose is done.
- **Genetic or health data** isn't a separate "sensitive" class under DPDP, but treat it as high-risk: minimise it, use India-hosted storage, set short retention.
- **Children:** mothers will talk about their children, which is personal data of a child (under 18). Rule 10 requires **verifiable parental consent**. The mother is the parent, so her consent covers it, but we should record her saying she is the child's parent and guardian [U, legal read]. Avoid collecting the child's name or identifiers.
- **Call recording:** no Indian statute explicitly requires both parties to consent. Recording by a participant is generally treated as lawful, but privacy (Puttaswamy) and DPDP make **announcing it and getting consent** the norm. [U, general legal knowledge] Always announce recording at the start of the call.

---

## Where we'll get stuck, ranked
1. **WhatsApp Business verification plus the ₹999/month WABA**: days to weeks. *Start now. Keep manual WhatsApp until it's approved.*
2. **A2P declaration for AI outbound calls** (since Sep 2026): the process through Plivo is unknown. *Email Plivo support this week.* Until it's sorted, use outbound only for human-initiated or scheduled calls, or rely on inbound and WhatsApp.
3. **No domestic SMS on Plivo.** *Drop SMS, or add a DLT vendor later.*
4. **7-day explicit consent for periodic service callbacks.** *Renew consent verbally on every call, keep logs, schedule the next call.*
5. **WhatsApp calling permission mechanics:** 7-day temporary permission, 4 missed calls revokes it. *Ask for "permanent" permission, schedule calls, prefer letting her call us.*
6. **Pickup on unknown 080 numbers / spam tagging.** *Send a WhatsApp heads-up before each call.*
7. **Media anchoring:** counsellors and conference participants must be on Indian numbers.
8. **Same number for voice and WhatsApp:** to be confirmed with Plivo.
9. **Whether a free pilot counts as "commercial communication":** get a legal opinion. Assume yes.

## Consents to capture (script lines, recorded)
1. **Identity & AI disclosure:** "Namaste, I'm SHISHU, an AI health-education assistant from [Company]. I'm not a doctor. A counsellor [is here / can join] anytime."
2. **Recording:** "This call is recorded and transcribed so our medical team can check the quality of information. Is that okay?" → *yes required, otherwise end or switch to a human.*
3. **Purpose & data (DPDP notice):** "I'll ask about health conditions in your family to explain inherited risks. We use this only for this education programme, store it securely in India, and delete it within [X] days. You can stop anytime or ask us to delete it by calling or messaging this number."
4. **Child data / parental consent:** "When you tell me about your child, can you confirm you are the child's mother or guardian and agree to share this?"
5. **Sensitive-topic opt-out:** "You can skip any question."
6. **WhatsApp opt-in:** "Can we send you a summary and next steps on WhatsApp? Which number?" Read the number back and get a "yes".
7. **Callback consent (TCCCPR explicit, 7 days):** "Can we call you back on this number in the next week? What day and time suits you?" Ask again on every call.
8. **WhatsApp call permission:** "I'll send a request on WhatsApp so we can call you there. Please tap 'Always allow' if you're comfortable."
9. **Withdrawal:** "Reply STOP or tell us anytime and we'll stop contacting you."
Log the timestamp, number, call UUID and audio offset for each consent.

## Pilot cost estimate (50 mothers, 2 months, excl. 18% GST)
Assumptions: 1 × 20-minute first call each, 1 × 10-minute callback each, 3 outbound follow-up messages each plus ~3 replies each.

| Item | Phone-only (WhatsApp sent by hand) | Full Plivo (PSTN first call + WhatsApp messages + WhatsApp callback) |
|---|---|---|
| Number ₹200 × 2 mo | ₹400 | ₹400 |
| First calls 1,000 min × ₹0.38 (if inbound on WhatsApp: × ₹0.45 = ₹450) | ₹380 | ₹380 |
| Callbacks 500 min: PSTN ₹0.38 / WhatsApp outbound ₹0.75 | ₹190 | ₹375 |
| WABA fee ₹999 × 2 | – | ₹1,998 |
| 150 outbound messages: Utility ₹0.115 + Plivo ₹0.10 (worst case, all templates) | – | ₹32 |
| 150 inbound messages × ₹0.10 Plivo fee | – | ₹15 |
| Recording, streaming, conferencing | ₹0 | ₹0 |
| **Total** | **≈ ₹970 (≈ ₹1,145 with GST)** | **≈ ₹3,200 (≈ ₹3,780 with GST)** |

The 30-second pulse adds at most about ₹0.19 per call, which is negligible. If the 1,000 free service messages apply and all messages go inside the window, outbound message cost drops to the ₹0.10 fee only. Not included: Plivo noise cancellation (₹0.12/min ≈ ₹180) and SMS (not available).

## Sources
- Plivo voice India pricing: https://www.plivo.com/voice/pricing/in/ [V]
- Plivo SMS India pricing (domestic not supported): https://www.plivo.com/sms/pricing/in/ [V]
- Plivo WhatsApp India pricing and fees: https://www.plivo.com/whatsapp/pricing/in/ [V]
- Plivo WhatsApp call pricing: https://www.plivo.com/whatsapp-call/pricing/in/ [V]
- Plivo India calling regulations: https://plivo.com/docs/voice/concepts/india-calling [V]
- Plivo India KYC: https://plivo.com/docs/numbers/rent-india-numbers ; https://old-support.plivo.com/hc/en-us/articles/36749835402905 [V]
- Plivo 140/160 provisioning: https://www.plivo.com/docs/voice/concepts/140-series-provisioning ; https://plivo.com/docs/voice/concepts/160-series-provisioning [V]
- Plivo WhatsApp calling: https://www.plivo.com/docs/messaging/concepts/whatsapp/whatsapp-calling [V]
- Plivo Stream XML: https://plivo.com/docs/voice/xml/audio-streaming [V]
- TRAI TCCCPR 2nd Amendment 2025: https://www.trai.gov.in/sites/default/files/2025-02/Regulation_12022025_0.pdf ; PIB https://www.pib.gov.in/PressReleasePage.aspx?PRID=2102413 [V]
- TRAI 1600-series Direction 19 Nov 2025: https://www.trai.gov.in/sites/default/files/2025-11/Direction_19112025_0.pdf [V]
- TRAI 3rd Amendment 2026 (A2P): https://trai.gov.in/sites/default/files/2026-09/PR_No119of2026.pdf ; https://trai.gov.in/sites/default/files/2026-09/Regulation_18092026.pdf [V]
- Meta call permissions (mirror; official page returned 403): https://whatsapp-docs.kap.so/documentation/whatsapp/calling/user-call-permissions [S] (official: https://developers.facebook.com/docs/whatsapp/cloud-api/calling/user-call-permissions)
- Meta INR rate card Oct 2026: https://xobito.com/downloads/whatsapp-rate-cards-october-2026/whatsapp-inr-rates-and-volume-tiers-oct-2026.pdf [S]
- WhatsApp Business Policy: https://whatsappbusiness.com/policy/ [S via excerpt]
- DPDP Rules 2025: https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf ; Rule 10 https://dpdprules.org/rules/10 [S]
- Calling time bands: https://telleo.ai/blog/trai-dlt-rules-ai-calling/ [S]

## Ask Plivo support in writing
1. How do we file the TRAI A2P declaration for AI-voice outbound calls from an 080 number?
2. Can the same rented 080 number be both our voice number and our WhatsApp number?
3. What is the default concurrent-channel limit?
4. Is there any DLT requirement for landline-series service calls?
