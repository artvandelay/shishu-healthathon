"""Seed two completed clinical calls when the database has none."""

import array
import math
import os
import shutil
import subprocess
import wave

import db

SAMPLE_MP4 = (
    "/Users/jigar/projects/0n-going/SHISHU-healthathon-koita/"
    "data/SHISHU Call/audio/Sample call recording-1Nov_Part 1.mp4"
)

_DOCTOR_SHILPA = """### Patient Profile & Chief Complaint
Mother: Shilpa. Child: Aarav, 7-year-old boy. This was a pre-consultation screening, not an acute emergency visit. The history a doctor should see first is neonatal jaundice treated with phototherapy, knee swelling for 5–7 days after falls, high blood pressure and thyroid disease that continued after delivery, and a previous son who died at 5 months with fits.

### Positive Red Flags & NRROID Gate Alerts
- Neonatal jaundice treated under blue light (phototherapy) for 3 days.
- Knee swelling after falls, lasting 5–7 days, then settling on its own. No excess bleeding and no easy bruising reported with this.
- High blood pressure and thyroid disease in pregnancy. Mother still takes medicines for both after delivery.
- Previous male infant died at 5 months. Mother reports fits and seizures as the cause.

### Birth & Neonatal History
Born at 9 months by normal vaginal delivery. Not admitted to NICU. No shock after birth. Jaundice treated with phototherapy for 3 days. Umbilical cord fell within 3 weeks. Hearing and vision were not tested at birth. No cleft of the lip or palate.

### Development, Growth & Physical Features
Sat, walked, and spoke on time compared with other children. No loss of a skill he already had. Abdomen is not getting bigger. No doctor report of an enlarged liver or intestine. Limbs look proportionate. Fingers are present. No deep wound in childhood. No pain in the hands or legs. Hair is black, not silver. Mother does not report swelling of the body or an atypical genital appearance.

### Systemic, Infections & Hematology
No repeated hospital admissions for breathing trouble. No blood transfusion. Does not bleed more than expected after a fall. Does not bruise easily. Urine odor is not unusual. Does not vomit often when sick. No known kidney disease.

### Antenatal (Pregnancy) History
High blood pressure and thyroid disease during pregnancy. Both continued after birth and still need medicines. Pregnancy ultrasound about every 3 months, reported normal. No blood transfusion in pregnancy. Delivery was uncomplicated. Thalassemia testing was not done for mother or father.

### Family & Pedigree History
Mother and father were not related before marriage. A previous son died at 5 months with fits and seizures. No family report of blindness, deafness, childhood cancer, blood transfusion, excess bleeding, heart disease, or kidney disease. High blood pressure is present in the in-laws. No family pattern of frequent infections with itchy skin.

### Pertinent Negatives
No NICU stay, no neonatal shock, no developmental regression, no organomegaly, no transfusion, no easy bruising, no consanguinity, and no drowsiness, vomiting, or fits when a meal is skipped.
"""

_MOTHER_SHILPA = """अगली बार जब आप डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से मिलें, तो ये सवाल ज़रूर पूछें:

1. जन्म के समय बच्चे को पीलिया के लिए 3 दिन नीली रोशनी (phototherapy) में रखना पड़ा था और सुनने-देखने की जाँच नहीं हुई थी — क्या अभी कोई जाँच ज़रूरी है?
2. गिरने पर बच्चे के घुटने में सूजन आ जाती है जो 5–7 दिन में ठीक होती है — कृपया देखकर बताएँ कि क्या इसके लिए खून या जोड़ों की कोई जाँच करवानी चाहिए?
3. प्रेग्नेंसी के समय से मुझे हाई बीपी और थायरॉइड की दवाई चल रही है और हमारा थैलेसीमिया टेस्ट नहीं हुआ है — मेरी और बच्चे की सेहत के लिए अभी कौन-सी जाँच करवानी चाहिए?
4. हमारे परिवार में पहले एक बच्चा 5 महीने की उम्र में दौरे (फिट्स) से गुजर गया था — क्या इस बच्चे या आगे की देखभाल के लिए हमें डॉक्टर से कोई विशेष सलाह या जाँच पूछनी चाहिए?

नोट: ये सवाल आपकी अगली विज़िट में डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से चर्चा के लिए हैं। यह किसी बीमारी की पुष्टि (डायग्नोसिस) नहीं है — केवल डॉक्टर ही जाँच करके सही सलाह दे सकते हैं।
"""

_DOCTOR_GEETA = """### Patient Profile & Chief Complaint
Mother: Geeta. Child: Ananya, 6-year-old girl. Screening visit for a routine pre-consultation history. Mother did not describe a current illness. Growth and milestones are reported as normal. No clinical red flag was recorded on this call.

### Positive Red Flags & NRROID Gate Alerts
None recorded. Transient high blood pressure in pregnancy resolved after delivery and is not flagged. Thalassemia testing was not done; mother had not heard of the test. That gap is noted for the doctor and is not marked as a positive disease flag.

### Birth & Neonatal History
Hospital birth at 9 months by normal vaginal delivery. No NICU admission. No jaundice needing blue light. No infection or shock in the first month. Umbilical cord did not bleed heavily and fell within 3 weeks. Mother does not see a BCG mark on the upper arm and asked what BCG means. Hearing and vision were not tested at birth. No cleft lip or palate. No shock later in childhood.

### Development, Growth & Physical Features
Did not sit, walk, or speak late. No loss of a previously learned skill. No known heart problem in pregnancy or at birth. Lips and palms do not turn blue. Abdomen is not enlarging. No report of a large liver or spleen. Body proportions look normal. Fingers and thumbs are present. Growth is like other children. Hair is black, not silver. No swelling or lumps since birth. Genital appearance reported as expected.

### Systemic, Infections & Hematology
No blood transfusion. No abnormal blood counts reported. No knee swelling after falls. No excess bleeding when a milk tooth fell. No easy blue spots or bruising. No repeated hospital admissions. No repeated infections. Urine and stool odor not unusual. No sudden collapse after a previously well newborn period. No recurrent joint or limb pain needing hospital care. No unusual sleepiness, repeated vomiting, or shaking when unwell. No known kidney disease.

### Antenatal (Pregnancy) History
High blood pressure during pregnancy only. It resolved after delivery and medicines were not continued. No diabetes or thyroid disease reported. Pregnancy scans were done and were normal. No blood transfusion in pregnancy. Delivery without a reported problem. Thalassemia test was not done. Mother asked what thalassemia is and said she was not aware of the test.

### Family & Pedigree History
Parents were not related before marriage. No loss of a child. No infant death under 1 year in the family. No family seizures, late milestones, blindness, deafness, transfusion need, bleeding tendency, or body lumps. No known heart or kidney disease. Childhood cancer in the family: mother does not know. High BP, cholesterol, or stroke at a young age: mother does not know. No known pattern of excess bleeding with frequent infections or itchy dry skin.

### Pertinent Negatives
Term hospital birth, no NICU, no phototherapy, no bleeding or joint swelling, no regression, no organomegaly, no transfusion, no consanguinity, no infant death, and no metabolic fasting crisis.
"""

_MOTHER_GEETA = """अगली बार जब आप डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से मिलें, तो ये सवाल ज़रूर पूछें:

1. बच्ची के हाथ पर BCG के टीके का निशान नहीं दिख रहा है और जन्म पर सुनने-देखने की जाँच नहीं हुई थी — कृपया टीकाकरण कार्ड देखकर बताएँ कि क्या कोई टीका या सामान्य जाँच बाकी है?
2. क्या मुझे और मेरे पति को थैलेसीमिया या खून की किसी सामान्य जाँच के बारे में डॉक्टर से सलाह लेनी चाहिए?
3. मेरी 6 साल की बच्ची की उम्र के हिसाब से उसका वज़न, लंबाई और खान-पान ठीक है या हमें किन बातों का ध्यान रखना चाहिए?

नोट: ये सवाल आपकी अगली विज़िट में डॉक्टर या स्वास्थ्य कार्यकर्ता (ASHA/ANM दीदी) से चर्चा के लिए हैं। यह किसी बीमारी की पुष्टि (डायग्नोसिस) नहीं है — केवल डॉक्टर ही जाँच करके सही सलाह दे सकते हैं।
"""

_TURNS_SHILPA = [
    ("assistant", "Namaste, main Shishu hoon. Kripya apna naam bataiye. (Hello, my name is Shishu. Please tell me your name.)"),
    ("user", "Mera naam Shilpa hai. (My name is Shilpa.)"),
    ("assistant", "Shilpa ji, main aapke bachche aur parivaar ke baare mein kuch sawal poochungi. Haan ya naa bol sakti hain. Jo na pata ho, woh theek hai. (I will ask about your child and your family history. Yes or no is fine. If you do not know, I will go to the next question.)"),
    ("user", "Theek hai. (Okay.)"),
    ("assistant", "Kya aapke bachche hain? (Do you have any children?)"),
    ("user", "Haan, mera 7 saal ka beta hai. Uska naam Aarav hai. (Yes, I have a seven-year-old son. His name is Aarav.)"),
    ("assistant", "Aarav normal delivery se paida hua tha ya operation se? 9 mahine mein hua tha ya usse pehle? (Was he born by normal delivery or caesarean? At nine months or earlier?)"),
    ("user", "Normal delivery, 9 mahine mein. (Normal. At nine months.)"),
    ("assistant", "Kya use NICU mein rakhna pada? Janam ke baad peeliya ke liye neeli roshni (blue light) par raha? (Did he need NICU? Was he placed on a blue light for jaundice?)"),
    ("user", "NICU nahi. Haan, peeliya ke liye 3 din neeli roshni par raha. (No NICU. Yes, blue light for three days.)"),
    ("assistant", "Janam ke baad jhatka ya shock to nahi hua? Baithna, chalna, bolna doosre bachchon ki tarah samay par hua? Koi cheez jo pehle karta tha, ab nahi karta? (Any shock after birth? Did he sit, walk, and speak on time? Any skill he lost?)"),
    ("user", "Shock nahi hua. Sab samay par seekha. Aisi koi baat nahi. (No shock. He learned on time. Nothing like that.)"),
    ("assistant", "Janam par sunne aur dekhne ki jaanch hui thi? Hoth ya taalu kata hua tha? (Hearing and sight tested at birth? A cut on the mouth?)"),
    ("user", "Jaanch nahi hui. Hoth kata nahi tha. (Not tested. No cut on the mouth.)"),
    ("assistant", "Saans ki takleef se baar-baar hospital? Pet dheere-dheere bada ho raha hai? Doctor ne badi aant ya liver bataya? (Repeated admissions for breathing? Tummy getting bigger? Large intestine or liver?)"),
    ("user", "Nahi, saans ke liye baar-baar nahi. Pet bada nahi ho raha. Doctor ne aisa nahi kaha. (No repeated breathing admissions. Tummy is not bigger. No.)"),
    ("assistant", "Khoon chadhane ki zaroorat padi? Girne par zyada khoon bahaa? Naal girne mein 3 hafte se zyada lage? (Any blood transfusion? More bleeding after a fall? Did the cord take more than three weeks to fall?)"),
    ("user", "Khoon nahi chadha. Girne par zyada khoon nahi bahaa. Naal 3 hafte se pehle gir gayi. (No transfusion. He did not bleed much. The cord did not take more than three weeks.)"),
    ("assistant", "Girne par ghutne mein sujan hoti hai? Kitne din rehti hai? Apne aap theek hoti hai? (Swelling in the knees when he falls? How long? Does it heal by itself?)"),
    ("user", "Haan, sujan hoti hai. 5 se 7 din rehti hai, phir apne aap theek ho jaati hai. (Yes. Five to seven days. It heals by itself.)"),
    ("assistant", "Aasani se neele nishan padte hain? Shareer par sujan? Baar-baar hospital mein bharti? Peshab ki gandh alag? (Easy bruises? Body swelling? Repeated admissions? Unusual urine smell?)"),
    ("user", "Nahi. Nishan aasani se nahi padte, sujan nahi, baar-baar hospital nahi, peshab ki gandh alag nahi. (No bruises, no body swelling, no repeated admissions, urine smell is not different.)"),
    ("assistant", "Haath-pair shareer se chhote lagte hain? Ungliyan chhoti ya nahi hain? Bachpan mein gehra ghav? Haath-pair mein dard? Baal silver jaise? Bimaar hone par baar-baar ulti? Gurde ki bimari? (Short limbs or fingers? Deep wound? Limb pain? Silver hair? Vomiting when sick? Kidney disease?)"),
    ("user", "Nahi. Sab normal hai. Baal kaale hain. Ulti baar-baar nahi hoti. Gurde ki bimari nahi hai. (No. Hair is black. No frequent vomiting. No kidney disease.)"),
    ("assistant", "Pregnancy mein high BP, sugar, ya thyroid hua tha? Delivery ke baad dawai band hui ya ab bhi chal rahi hai? (High blood pressure, diabetes, or thyroid in pregnancy? Still on medicines after birth?)"),
    ("user", "Haan, high BP aur thyroid tha. Ab bhi dawai leni padti hai. (Yes, high blood pressure and thyroid. I still need medicines.)"),
    ("assistant", "Aap aur pati ka thalassemia test hua? Pregnancy mein scan normal tha? Garbh mein khoon chadhaya? Delivery mein koi dikkat? (Thalassemia test? Scans normal? Blood in pregnancy? Problem in delivery?)"),
    ("user", "Thalassemia test nahi hua. Har 3 mahine scan hota tha, normal tha. Khoon nahi chadha. Delivery normal rahi. (No thalassemia test. Scan every three months, normal. No transfusion. Normal delivery.)"),
    ("assistant", "Ab parivaar ki baat. Kya aap aur aapke pati shaadi se pehle rishtedaar the? Kya pehle koi bachcha guzar gaya? (Were you and your husband relatives before marriage? Has a child died?)"),
    ("user", "Nahi, hum rishtedaar nahi the. Ek beta tha, woh 5 mahine mein guzar gaya. Use jhatke aur seizures aate the. (No, we were not relatives. I had a son and he passed away at five months. He had fits and seizures.)"),
    ("assistant", "Parivaar mein kisi ko dekhne ki dikkat, kam sunai dena, khoon chadhane ki zaroorat, zyada khoon bahna, bachpan ka cancer, dil, ya gurde ki bimari? (Blindness, hearing loss, transfusions, excess bleeding, childhood cancer, heart, or kidney disease in the family?)"),
    ("user", "Nahi, inmein se koi baat nahi. (No.)"),
    ("assistant", "Parivaar mein BP, cholesterol, ya stroke? Kis ko? Zyada khoon bahna, baar-baar infection, ya khujli wali twacha? (BP, cholesterol, or stroke in the family? Who? Excess bleeding, frequent infections, or itchy skin?)"),
    ("user", "Parivaar mein BP hai, mere sasural mein. Baaki aisi koi baat nahi. (There is a BP problem in the family, my in-laws. Nothing else like that.)"),
    ("assistant", "Is call ke baad aapko ek summary milegi jo aap doctor ko dikha sakti hain. Dhanyavaad. (You will get a summary after this call to show your doctor. Thank you.)"),
    ("user", "Theek hai. Dhanyavaad. (Okay. Thank you.)"),
]

_FINDINGS_SHILPA = [
    ("profile_mother_name", "info", "Shilpa", False),
    ("profile_child_name", "info", "Aarav", False),
    ("profile_child_age", "info", "7 years", False),
    ("profile_child_gender", "info", "male", False),
    ("chief_complaint", "info", "Pre-consultation screen. No single acute complaint stated.", False),
    ("birth_delivery", "negative", "Term normal vaginal delivery at 9 months.", False),
    ("birth_nicu", "negative", "No NICU admission.", False),
    ("birth_phototherapy", "positive", "Neonatal jaundice treated with blue-light phototherapy for 3 days.", True),
    ("birth_shock", "negative", "No shock after birth.", False),
    ("milestones", "negative", "Sat, walked, and spoke on time. No regression.", False),
    ("hearing_vision", "info", "Hearing and vision not tested at birth.", False),
    ("cleft", "negative", "No cleft lip or palate.", False),
    ("infections", "negative", "No repeated admissions for breathing problems.", False),
    ("organomegaly", "negative", "Abdomen not enlarging. No reported large liver or intestine.", False),
    ("transfusion", "negative", "No blood transfusion.", False),
    ("bleeding", "negative", "Does not bleed more than expected after falls.", False),
    ("umbilical_cord", "negative", "Cord fell within 3 weeks.", False),
    ("bleeding_joints", "positive", "Knee swelling after falls lasting 5-7 days, then resolves on its own.", True),
    ("bruising", "negative", "Does not bruise easily.", False),
    ("genitalia", "negative", "Genital appearance reported as expected.", False),
    ("body_swelling", "negative", "No body swelling.", False),
    ("urine_odor", "negative", "Urine odor not unusual.", False),
    ("body_habitus", "negative", "Limbs and fingers proportionate. No deep childhood wound. No limb pain. Hair black.", False),
    ("metabolic_fasting", "negative", "No frequent vomiting when sick.", False),
    ("kidney", "negative", "No known kidney disease.", False),
    ("pregnancy", "positive", "High BP and thyroid in pregnancy, still on medicines after delivery.", True),
    ("thalassemia_screen", "info", "Thalassemia test not done for mother or father.", False),
    ("pregnancy_scans", "negative", "Scans about every 3 months, reported normal. No transfusion in pregnancy. Uncomplicated delivery.", False),
    ("consanguinity", "negative", "Parents were not related before marriage.", False),
    ("family_infant_death", "positive", "Previous male infant died at 5 months due to fits and seizures.", True),
    ("family_history", "info", "High blood pressure in the in-laws. No blindness, deafness, transfusion dependence, excess bleeding, childhood cancer, heart disease, kidney disease, or itchy-skin infection pattern.", False),
]

_TURNS_GEETA = [
    ("assistant", "Namaste, main Shishu hoon. Kripya apna naam bataiye. (Namaste, I am Shishu. Can you tell me your name?)"),
    ("user", "Mera naam Geeta hai. (My name is Geeta.)"),
    ("assistant", "Geeta ji, kya aapki shaadi hui hai? Bachche hain? (Are you married? Do you have children?)"),
    ("user", "Haan, shaadi hui hai. Ek beti hai, Ananya, 6 saal ki. (Yes. I have a daughter. She is 6 years old.)"),
    ("assistant", "Ananya normal delivery se hui ya operation se? 9 mahine mein? Ghar par ya hospital mein? (Normal delivery or operation? At 9 months? Home or hospital?)"),
    ("user", "Normal delivery, 9 mahine mein, hospital mein. (Normal delivery. At 9 months. In the hospital.)"),
    ("assistant", "NICU ki zaroorat padi? Pehle mahine mein peeliya ke liye neeli roshni? Infection ya shock? (NICU? Jaundice with blue light in the first month? Infection or shock?)"),
    ("user", "Nahi. NICU nahi, neeli roshni nahi, infection ya shock nahi. (No. No NICU, no blue light, no infection or shock.)"),
    ("assistant", "Naal girte waqt zyada khoon baha? Naal girne mein 3 hafte se zyada lage? BCG ke baad haath par asadharan nishan ya peep? (Heavy bleed when the cord fell? More than 3 weeks? Unusual mark or pus after BCG?)"),
    ("user", "Naal se zyada khoon nahi baha, aur 3 hafte se pehle gir gayi. BCG kya hota hai? Haath par koi nishan nahi dikhta. (No heavy cord bleeding, and it did not take more than 3 weeks. What is BCG? I do not see a mark on her arm.)"),
    ("assistant", "BCG janam ke baad lagayi jaane wali vaccine hai. Aksar upar ki baah par ek chhota nishan reh jaata hai. Sunne aur dekhne ki jaanch hui? Hoth ya taalu kata tha? Kabhi jhatka hua? (BCG is a vaccine given after birth. It often leaves a mark on the upper arm. Hearing and sight tested? Cleft? Any shock?)"),
    ("user", "Jaanch nahi hui. Hoth kata nahi tha. Jhatka nahi hua. (Not tested. No cleft. No shock.)"),
    ("assistant", "Baithna, chalna, bolna der se hua? Koi cheez jo pehle karti thi, ab nahi karti? Dil ki baat pregnancy ya janam par? Hoth ya hatheli neeli hoti hai? (Late to sit, walk, or speak? Any lost skill? Heart problem? Blue lips or palms?)"),
    ("user", "Nahi, der se nahi hua. Koi skill khoi nahi. Dil ki baat nahi. Neela nahi hota. (No, not late. Nothing lost. No heart problem. Not blue.)"),
    ("assistant", "Pet bada ho raha hai ya liver/spleen bada bataya? Khoon chadha? Blood count oonchi ya neechi? Girne par ghutne mein sujan jo kuch din mein theek ho? Doodh ka daant girte waqt zyada khoon? Aasani se neele dhabbe? (Big tummy or liver/spleen? Transfusion? Abnormal counts? Knee swelling after falls? Extra bleeding when a tooth fell? Easy blue spots?)"),
    ("user", "Nahi. Pet normal hai. Khoon nahi chadha. Count ki baat nahi. Ghutne ki sujan nahi. Daant girte waqt zyada khoon nahi. Neele dhabbe nahi. (No. No transfusion, no knee swelling, no extra bleeding, no blue spots.)"),
    ("assistant", "Shareer par sujan ya gaanth? Baar-baar hospital ya infection? Peshab ya potty ki gandh alag? Janam par theek thi, phir achanak bahut beemar hokar emergency? Haath-pair chhote? Ungli ya angutha chhota? Doosre bachchon ki tarah badh rahi hai? Jodon mein dard? Baal silver? Bimaar hone par neend, ulti, ya jhatke? Gurde ki bimari? (Lumps? Repeated admissions or infections? Unusual urine or stool smell? Sudden emergency illness? Short limbs or fingers? Growing normally? Joint pain? Silver hair? Sleepiness, vomiting, or shaking when sick? Kidney disease?)"),
    ("user", "Nahi. Sab theek hai. Shareer normal dikhta hai. Doosre bachchon ki tarah badh rahi hai. Baal kaale hain. Aisi koi beemari nahi. Gurde ki bimari nahi. (No. Everything is fine. She looks normal and is growing like other children. Hair is black. Nothing like that. No kidney disease.)"),
    ("assistant", "Pregnancy mein high BP, sugar, ya thyroid? Delivery ke baad bhi raha? (High BP, diabetes, or thyroid in pregnancy? Did it continue after delivery?)"),
    ("user", "High BP tha. Delivery ke baad theek ho gaya. (I had high BP. It was fine after delivery.)"),
    ("assistant", "Thalassemia test hua? Scan hua, normal tha? Garbh mein khoon chadha? Delivery bina dikkat ke hui? (Thalassemia test? Scan normal? Blood in pregnancy? Delivery without a problem?)"),
    ("user", "Thalassemia kya hai? Mujhe iska pata nahi tha, test nahi hua. Scan hua tha, normal tha. Khoon nahi chadha. Delivery theek rahi. (What is thalassemia? I was not aware of it, so no test. Scan was normal. No blood. Delivery was fine.)"),
    ("assistant", "Thalassemia mein khoon theek se nahi banta, aur kuch logon ko baar-baar khoon chadhane ki zaroorat padti hai. Test pehle se ho sakta hai. Doctor hi bataenge ki aapke liye zaroorat hai ya nahi. Ab parivaar: kya aap aur pati shaadi se pehle rishtedaar the? Koi bachcha khoya? Ek saal se kam umar mein kisi bachche ki mrityu? Parivaar mein jhatke? (Thalassemia is a condition where blood does not form normally and some people need transfusions. A test can be done in advance. The doctor will say if you need it. Were you related before marriage? Any child lost? Any death under one year? Seizures in the family?)"),
    ("user", "Nahi, hum rishtedaar nahi the. Koi bachcha nahi khoya. Ek saal se kam umar mein koi mrityu nahi. Jhatke nahi. (No. We were not relatives. I have not lost a child. No death under one year. No seizures.)"),
    ("assistant", "Parivaar mein der se baithna-chalna-bolna, andhaapan, kam sunai dena, khoon chadhna, zyada khoon bahna, bachpan ka cancer, dil, gurde, jawan umar mein BP/cholesterol/stroke, infection ke saath khujli, ya shareer par gaanth? (Late milestones, blindness, deafness, transfusions, bleeding, childhood cancer, heart, kidney, young BP/cholesterol/stroke, itchy infections, or lumps in the family?)"),
    ("user", "Nahi, jitna mujhe pata hai. Bachpan ka cancer mujhe nahi pata. Jawan umar ka BP ya stroke bhi nahi pata. Baaki nahi. (No, as far as I know. I do not know about childhood cancer, or about high BP or stroke at a young age. Otherwise no.)"),
    ("assistant", "Kya aap doctor ke liye kuch aur batana chahengi? (Do you want to say anything else?)"),
    ("user", "Nahi. (No.)"),
    ("assistant", "Jawaab dene ke liye dhanyavaad. Aapne jo bataya, woh doctor ke liye note ho gaya hai. Aage agar aaj jo humne poocha usmein se kuch dikhe, to health worker ko bata dena. (Thank you. This is noted for your doctor. If you later see something we asked about today, tell your health worker.)"),
]

_FINDINGS_GEETA = [
    ("profile_mother_name", "info", "Geeta", False),
    ("profile_child_name", "info", "Ananya", False),
    ("profile_child_age", "info", "6 years", False),
    ("profile_child_gender", "info", "female", False),
    ("chief_complaint", "info", "Routine pre-consultation screen. No current illness described.", False),
    ("birth_delivery", "negative", "Term normal vaginal delivery in hospital at 9 months.", False),
    ("birth_nicu", "negative", "No NICU admission. No infection or shock in the first month.", False),
    ("birth_phototherapy", "negative", "No neonatal jaundice needing blue light.", False),
    ("umbilical_cord", "negative", "Cord did not bleed heavily and fell within 3 weeks.", False),
    ("bcg", "info", "Mother does not see a BCG mark and asked what BCG is. No pus or swelling described.", False),
    ("hearing_vision", "info", "Hearing and vision not tested at birth.", False),
    ("cleft", "negative", "No cleft lip or palate.", False),
    ("milestones", "negative", "Sat, walked, and spoke on time. No regression. Growing like peers.", False),
    ("heart", "negative", "No known heart problem. Lips and palms do not turn blue.", False),
    ("organomegaly", "negative", "Abdomen not enlarging. No large liver or spleen reported.", False),
    ("transfusion", "negative", "No blood transfusion. No abnormal blood counts reported.", False),
    ("bleeding_joints", "negative", "No knee swelling after falls.", False),
    ("bleeding", "negative", "No excess bleeding when a milk tooth fell. No easy bruising.", False),
    ("genitalia", "negative", "Genital appearance reported as expected.", False),
    ("body_swelling", "negative", "No swelling or lumps since birth.", False),
    ("infections", "negative", "No repeated hospital admissions or infections.", False),
    ("urine_odor", "negative", "Urine and stool odor not unusual.", False),
    ("metabolic_fasting", "negative", "No sudden emergency collapse. No sleepiness, repeated vomiting, or shaking when unwell.", False),
    ("body_habitus", "negative", "Proportions normal. Fingers and thumbs present. No recurrent joint pain. Hair black.", False),
    ("kidney", "negative", "No known kidney disease.", False),
    ("pregnancy", "info", "Transient high BP in pregnancy, resolved after delivery. No diabetes or thyroid reported.", False),
    ("thalassemia_screen", "info", "Thalassemia test not done. Mother had not heard of thalassemia.", False),
    ("pregnancy_scans", "negative", "Pregnancy scans done and reported normal. No transfusion. Delivery without a reported problem.", False),
    ("consanguinity", "negative", "Parents were not related before marriage.", False),
    ("family_infant_death", "negative", "No child loss. No infant death under 1 year. No family seizures.", False),
    ("family_history", "info", "No known late milestones, blindness, deafness, transfusion need, bleeding tendency, heart disease, kidney disease, or body lumps. Childhood cancer and young-age BP/cholesterol/stroke unknown to mother.", False),
]


def _wav_duration_sec(path: str) -> int:
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate() or 1
        return max(1, int(handle.getnframes() / rate))


def _write_placeholder_wav(path: str, seconds: int = 3) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rate = 24000
    count = rate * seconds
    samples = array.array(
        "h",
        (int(4000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(count)),
    )
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return seconds


def _convert_sample_or_placeholder(path: str) -> int:
    """Write call audio from the Nov 1 sample, or a 3-second wav if it is missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if shutil.which("ffmpeg") and os.path.isfile(SAMPLE_MP4):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                SAMPLE_MP4,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                path,
            ],
            check=True,
        )
        return _wav_duration_sec(path)
    return _write_placeholder_wav(path, 3)


def _seed_call(turns, findings, summary_doctor, summary_mother, use_sample_audio: bool) -> int:
    call_id = db.create_call("gemini")
    for role, text in turns:
        db.append_transcript(call_id, role, text)
    for domain, status, detail, is_red_flag in findings:
        db.record_finding_db(call_id, domain, status, detail, is_red_flag)
    audio_file = f"call_{call_id}.wav"
    audio_path = os.path.join(db.DATA_DIR, "audio", audio_file)
    if use_sample_audio:
        duration = _convert_sample_or_placeholder(audio_path)
    else:
        duration = _write_placeholder_wav(audio_path, 3)
    db.complete_call(call_id, duration, audio_file, summary_doctor, summary_mother)
    return call_id


def ensure_seeded() -> None:
    if db.list_calls():
        return
    _seed_call(_TURNS_SHILPA, _FINDINGS_SHILPA, _DOCTOR_SHILPA, _MOTHER_SHILPA, True)
    _seed_call(_TURNS_GEETA, _FINDINGS_GEETA, _DOCTOR_GEETA, _MOTHER_GEETA, False)
