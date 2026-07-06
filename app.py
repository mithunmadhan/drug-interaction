"""
India Drug Interaction Checker — Streamlit Web App

IMPORTANT: First-pass flagger for internal eval only.
           No clinical sign-off. NOT for clinical decision support.
           NON-COMMERCIAL / EVAL ONLY.
"""

import sqlite3, pathlib
from itertools import combinations

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drug Interaction Checker",
    page_icon="💊",
    layout="wide",
)

ROOT    = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "out"

# ── Data loading (cached — runs once on startup) ──────────────────────────────

import json
import os

def load_env_file():
    # Try using python-dotenv if installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Fallback: manual parsing from .env file if it exists in the workspace
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_env_file()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

@st.cache_resource
def load_data():
    db_path = OUT_DIR / "app_data.db"
    if not db_path.exists():
        return {}, {}, set(), {}, {}
        
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    
    # Load crosswalk
    cw = pd.read_sql_query("SELECT name_lower, rxcui FROM crosswalk", con)
    xwalk = dict(zip(cw["name_lower"], cw["rxcui"]))
    rxcui_to_name = {v: k.title() for k, v in xwalk.items()}

    # Load interactions
    rows = con.execute("SELECT rxcui_a, rxcui_b, note, category FROM interactions").fetchall()
    interaction_set   = {(a, b) for a, b, _, _ in rows}
    interaction_notes = {(a, b): (note, cat) for a, b, note, cat in rows}

    # Load medicines for brand index
    master = pd.read_sql_query("SELECT brand, ingredients FROM medicines", con)
    con.close()

    brand_index = {}
    for _, row in master.iterrows():
        bname = str(row.get("brand", "")).strip()
        if not bname:
            continue
        ings_str = row.get("ingredients")
        if not ings_str:
            continue
        try:
            ings = json.loads(ings_str) if isinstance(ings_str, str) else ings_str
        except Exception:
            continue
        if not isinstance(ings, list) or len(ings) == 0:
            continue
        ing_list = [
            {"ingredient": s.get("ingredient", ""), "rxcui": s.get("rxcui")}
            for s in ings
            if s.get("rxcui")
        ]
        if ing_list:
            brand_index[bname.lower()] = (bname, ing_list)

    return xwalk, rxcui_to_name, interaction_set, interaction_notes, brand_index


xwalk, rxcui_to_name, INTERACTION_SET, INTERACTION_NOTES, brand_index = load_data()

# ── Drug class expansions ─────────────────────────────────────────────────────
# Maps common class search terms → list of specific drug names in our crosswalk
# so "nitrates" correctly triggers sildenafil × isosorbide mononitrate etc.

CLASS_EXPANSIONS: dict[str, list[str]] = {
    "nitrates":          ["isosorbide mononitrate", "isosorbide dinitrate", "nitroglycerin"],
    "nitrate":           ["isosorbide mononitrate", "isosorbide dinitrate", "nitroglycerin"],
    "nsaids":            ["ibuprofen", "naproxen", "diclofenac", "aspirin", "celecoxib", "ketorolac", "piroxicam"],
    "nsaid":             ["ibuprofen", "naproxen", "diclofenac", "aspirin", "celecoxib", "ketorolac"],
    "beta blockers":     ["metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol", "nebivolol"],
    "beta-blockers":     ["metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol"],
    "statins":           ["simvastatin", "atorvastatin", "rosuvastatin", "lovastatin", "pravastatin"],
    "ssri":              ["sertraline", "fluoxetine", "paroxetine", "escitalopram", "citalopram"],
    "ssris":             ["sertraline", "fluoxetine", "paroxetine", "escitalopram", "citalopram"],
    "ace inhibitors":    ["enalapril", "lisinopril", "ramipril", "perindopril", "trandolapril"],
    "arbs":              ["losartan", "telmisartan", "valsartan", "olmesartan", "irbesartan"],
    "fluoroquinolones":  ["ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin"],
    "macrolides":        ["clarithromycin", "erythromycin", "azithromycin"],
    "antifungals":       ["fluconazole", "itraconazole", "ketoconazole", "voriconazole"],
    "azoles":            ["fluconazole", "itraconazole", "ketoconazole", "voriconazole"],
    "anticoagulants":    ["warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban", "dabigatran"],
    "benzodiazepines":   ["diazepam", "alprazolam", "clonazepam", "lorazepam", "midazolam"],
    "opioids":           ["morphine", "codeine", "tramadol", "fentanyl", "oxycodone", "methadone"],
    "maois":             ["phenelzine", "tranylcypromine", "isocarboxazid", "selegiline"],
    "sulfonylureas":     ["glibenclamide", "glipizide", "glimepiride", "gliclazide"],
    "aminoglycosides":   ["gentamicin", "amikacin", "tobramycin"],
    "antipsychotics":    ["haloperidol", "chlorpromazine", "quetiapine", "risperidone", "olanzapine"],
    "ppi":               ["omeprazole", "esomeprazole", "pantoprazole", "rabeprazole", "lansoprazole"],
    "ppis":              ["omeprazole", "esomeprazole", "pantoprazole", "rabeprazole", "lansoprazole"],
    "diuretics":         ["furosemide", "hydrochlorothiazide", "spironolactone", "torsemide"],
    "immunosuppressants":["cyclosporine", "tacrolimus", "azathioprine", "methotrexate"],
    "calcium channel blockers": ["amlodipine", "verapamil", "diltiazem", "nifedipine", "felodipine"],
}


def expand_class(name_lower: str) -> list[tuple[str, str]] | None:
    """If name matches a drug class, return list of (drug_name, rxcui) for its members."""
    members = CLASS_EXPANSIONS.get(name_lower)
    if not members:
        return None
    result = []
    seen = set()
    for m in members:
        rxcui = xwalk.get(m)
        if rxcui and rxcui not in seen:
            seen.add(rxcui)
            result.append((m.title(), rxcui))
    return result if result else None


# ── Search helpers ────────────────────────────────────────────────────────────

def search_drugs(query: str, limit: int = 12) -> list[dict]:
    """
    Return list of matches for a query string.
    Checks drug class expansions first, then brand names, then generics.
    Each result: {display, rxcuis: [(name, rxcui)], source}
    """
    q = query.strip().lower()
    if len(q) < 2:
        return []

    results = []
    seen_keys = set()

    # 0. Drug class expansion (exact match on class name)
    for class_name, members in CLASS_EXPANSIONS.items():
        if q in class_name or class_name.startswith(q):
            expanded = expand_class(class_name)
            if expanded:
                key = f"class:{class_name}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    ing_str = " + ".join(n for n, _ in expanded[:4])
                    results.append({
                        "display": f"{class_name.title()}  (expands to: {ing_str}…)",
                        "label":   class_name.title(),
                        "rxcuis":  expanded,
                        "source":  "class",
                    })

    # 1. Brand name match
    for bname_lower, (bname, ing_list) in brand_index.items():
        if q in bname_lower:
            key = bname_lower
            if key not in seen_keys:
                seen_keys.add(key)
                ing_str = " + ".join(i["ingredient"].title() for i in ing_list[:4])
                results.append({
                    "display": f"{bname}  ({ing_str})",
                    "label":   bname,
                    "rxcuis":  [(i["ingredient"].title(), i["rxcui"]) for i in ing_list],
                    "source":  "brand",
                })
            if len(results) >= limit:
                break

    # 2. Generic name prefix match from crosswalk
    for name, rxcui in xwalk.items():
        if name.startswith(q) and len(name) >= 3:
            key = f"generic:{rxcui}"
            if key not in seen_keys:
                seen_keys.add(key)
                results.append({
                    "display": name.title(),
                    "label":   name.title(),
                    "rxcuis":  [(name.title(), rxcui)],
                    "source":  "generic",
                })
            if len(results) >= limit:
                break

    return results[:limit]


def pair_key(a, b):
    return (a, b) if a < b else (b, a)


def check_medlist(rxcui_list: list[str]) -> list[dict]:
    flags = []
    for a, b in combinations(set(rxcui_list), 2):
        k = pair_key(a, b)
        if k in INTERACTION_SET:
            note, cat = INTERACTION_NOTES.get(k, ("", ""))
            flags.append({
                "rxcui_a":  k[0],
                "rxcui_b":  k[1],
                "name_a":   rxcui_to_name.get(k[0], k[0]),
                "name_b":   rxcui_to_name.get(k[1], k[1]),
                "note":     note,
                "category": cat,
            })
    return flags


# ── Clinical consequences by category ────────────────────────────────────────
# Plain-English description of what the patient may experience

CONSEQUENCES: dict[str, dict] = {
    "hypotension": {
        "problem":  "Dangerous drop in blood pressure",
        "symptoms": "Severe dizziness, fainting, rapid heartbeat, cold sweats. In serious cases: heart attack or stroke.",
        "action":   "Do NOT take together. This combination is contraindicated.",
    },
    "qt": {
        "problem":  "Abnormal heart rhythm (QT prolongation)",
        "symptoms": "Palpitations, dizziness, fainting, irregular pulse. Can progress to Torsades de Pointes — a life-threatening arrhythmia.",
        "action":   "Avoid combination. If unavoidable, monitor ECG closely.",
    },
    "serotonin": {
        "problem":  "Serotonin syndrome",
        "symptoms": "Agitation, confusion, rapid heart rate, high temperature, sweating, muscle twitching, tremor. Severe cases: seizures or death.",
        "action":   "Avoid combination. Seek emergency care immediately if symptoms appear.",
    },
    "bleeding": {
        "problem":  "Increased bleeding risk",
        "symptoms": "Unusual bruising, prolonged bleeding from cuts, blood in urine or stool, internal bleeding, severe GI haemorrhage.",
        "action":   "Use with extreme caution. Monitor for signs of bleeding. May need dose adjustment or alternative.",
    },
    "cyp3a4": {
        "problem":  "Drug level too high or too low in blood",
        "symptoms": "Varies by drug — toxicity symptoms (nausea, muscle pain, confusion) if level rises; treatment failure if level drops.",
        "action":   "Monitor drug levels and patient symptoms. Dose adjustment usually required.",
    },
    "cyp2c9": {
        "problem":  "Altered drug metabolism (CYP2C9 enzyme)",
        "symptoms": "Risk of bleeding (if warfarin rises) or hypoglycaemia (if sulfonylurea rises). Depends on specific drug pair.",
        "action":   "Monitor INR / blood glucose closely. Dose adjustment may be needed.",
    },
    "cyp2d6": {
        "problem":  "Altered drug metabolism (CYP2D6 enzyme)",
        "symptoms": "Drug toxicity or loss of effect depending on direction. E.g. metoprolol toxicity causing bradycardia, or codeine analgesia failure.",
        "action":   "Consider alternative drug or adjust dose with monitoring.",
    },
    "cyp2c19": {
        "problem":  "Altered drug metabolism (CYP2C19 enzyme)",
        "symptoms": "Reduced antiplatelet effect (clopidogrel) or elevated drug levels causing toxicity.",
        "action":   "Consider alternative PPI or antiplatelet agent. Monitor response.",
    },
    "cyp1a2": {
        "problem":  "Altered drug metabolism (CYP1A2 enzyme)",
        "symptoms": "Drug toxicity, e.g. theophylline toxicity: nausea, seizures, arrhythmia. Or clozapine toxicity: sedation, seizures.",
        "action":   "Monitor drug levels. Dose reduction of the affected drug often needed.",
    },
    "nephrotox": {
        "problem":  "Kidney damage (nephrotoxicity)",
        "symptoms": "Reduced urine output, swelling in legs/feet, fatigue, nausea, confusion. Can lead to acute kidney failure.",
        "action":   "Avoid combination if possible. Monitor kidney function (creatinine, urea) regularly.",
    },
    "hyperkalaemia": {
        "problem":  "Dangerously high potassium levels",
        "symptoms": "Muscle weakness, fatigue, tingling, irregular heartbeat. Severe: cardiac arrest.",
        "action":   "Monitor serum potassium levels. Avoid in patients with kidney disease.",
    },
    "myopathy": {
        "problem":  "Muscle damage (myopathy / rhabdomyolysis)",
        "symptoms": "Muscle pain, weakness, dark-coloured urine (from myoglobin). Severe rhabdomyolysis can cause kidney failure.",
        "action":   "Stop statin if unexplained muscle pain. This combination is usually contraindicated.",
    },
    "haematotox": {
        "problem":  "Bone marrow suppression",
        "symptoms": "Low blood counts — anaemia (fatigue, pallor), increased infection risk, easy bruising/bleeding.",
        "action":   "Avoid combination. Monitor full blood count regularly if unavoidable.",
    },
    "hepatotox": {
        "problem":  "Liver damage (hepatotoxicity)",
        "symptoms": "Nausea, jaundice (yellowing of skin/eyes), dark urine, upper right abdominal pain, fatigue.",
        "action":   "Avoid combination. Monitor liver function tests.",
    },
    "cns": {
        "problem":  "Excessive sedation / CNS depression",
        "symptoms": "Extreme drowsiness, confusion, impaired coordination, slowed breathing. Risk of respiratory arrest.",
        "action":   "Avoid driving or operating machinery. Use lowest effective doses. Risk is highest in elderly patients.",
    },
    "hypoglycaemia": {
        "problem":  "Dangerously low blood sugar",
        "symptoms": "Shakiness, sweating, confusion, fast heartbeat, hunger. Severe: loss of consciousness or seizures.",
        "action":   "Monitor blood glucose closely. Patient should carry fast-acting sugar.",
    },
    "pgp": {
        "problem":  "Drug transport disruption (P-glycoprotein)",
        "symptoms": "Elevated drug levels causing toxicity — e.g. digoxin toxicity: nausea, vision changes, arrhythmia.",
        "action":   "Monitor drug levels. Dose reduction of the affected drug usually needed.",
    },
    "cardio": {
        "problem":  "Cardiac conduction problem",
        "symptoms": "Slow heart rate (bradycardia), low blood pressure, dizziness, fatigue. Risk of heart block.",
        "action":   "Avoid combining two drugs that slow heart rate. Monitor pulse and ECG.",
    },
    "enzymatic": {
        "problem":  "Enzyme pathway blockage causing drug accumulation",
        "symptoms": "Toxicity of the accumulated drug — e.g. azathioprine toxicity: bone marrow suppression, severe infections.",
        "action":   "This combination is often contraindicated. Consult a specialist before use.",
    },
    "lithium": {
        "problem":  "Lithium toxicity",
        "symptoms": "Tremor, confusion, slurred speech, diarrhoea, nausea. Severe: seizures, kidney failure, coma.",
        "action":   "Monitor lithium levels closely. Adjust dose or use alternative drug.",
    },
    "absorption": {
        "problem":  "Drug absorption blocked",
        "symptoms": "Treatment failure — the antibiotic or drug is not absorbed properly, reducing its effectiveness.",
        "action":   "Separate doses by at least 2–4 hours. Take antibiotic first.",
    },
    "disulfiram": {
        "problem":  "Disulfiram-like reaction with alcohol",
        "symptoms": "Flushing, severe nausea, vomiting, headache, rapid heart rate, low blood pressure.",
        "action":   "Avoid alcohol completely while taking this drug.",
    },
    "drugcentral-critical": {
        "problem":  "Serious / potentially life-threatening interaction",
        "symptoms": "Depends on specific pair — may include cardiac events, severe toxicity, or life-threatening adverse effects.",
        "action":   "Consult a doctor or pharmacist before taking together. This is a critical interaction.",
    },
    "drugcentral-significant": {
        "problem":  "Clinically significant interaction",
        "symptoms": "May reduce drug effectiveness or increase risk of side effects. Clinical monitoring is needed.",
        "action":   "Inform your doctor about all medications you are taking.",
    },
    "openfda": {
        "problem":  "Known drug interaction (FDA label)",
        "symptoms": "See drug label for specific effects. Monitor for unusual symptoms.",
        "action":   "Consult a pharmacist or doctor before combining these drugs.",
    },
    "other": {
        "problem":  "Clinically relevant interaction",
        "symptoms": "May increase side effects or reduce effectiveness of one or both drugs.",
        "action":   "Consult a doctor or pharmacist.",
    },
}

def get_consequences(cat: str) -> dict:
    return CONSEQUENCES.get(cat, CONSEQUENCES["other"])


# ── Severity colour ───────────────────────────────────────────────────────────

CRITICAL_CATS = {"drugcentral-critical", "hypotension", "qt", "serotonin", "enzymatic"}
HIGH_CATS     = {
    "drugcentral-significant", "cyp3a4", "cyp2c9", "cyp2d6", "cyp2c19",
    "cyp1a2", "bleeding", "nephrotox", "haematotox", "hepatotox",
}

def severity_badge(cat: str) -> str:
    if cat in CRITICAL_CATS:
        return "🔴 CRITICAL"
    if cat in HIGH_CATS:
        return "orange badges 🟠 HIGH"
    return "🟡 MODERATE"

def severity_color(cat: str) -> str:
    if cat in CRITICAL_CATS:
        return "#FF4B4B"
    if cat in HIGH_CATS:
        return "#FFA500"
    return "#FFD700"

# ── RAG LLM Generator ─────────────────────────────────────────────────────────

def generate_ddi_explanation(drugs_added, flags):
    """
    Call Gemini 3.5 Flash using the Google GenAI SDK to generate a patient-friendly summary.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
            
    if not key:
        return "⚠️ **Gemini API Key missing.** Please set the `GEMINI_API_KEY` environment variable in your `.env` file or Streamlit secrets."

    if not GENAI_AVAILABLE:
        return "❌ **GenAI SDK missing.** Please run `pip install google-genai` to enable LLM summary generation."

    try:
        client = genai.Client(api_key=key)
        
        # Format the context from retrieved DDI flags
        context_parts = []
        for idx, f in enumerate(flags, 1):
            context_parts.append(
                f"{idx}. {f['name_a'].title()} + {f['name_b'].title()}: "
                f"{f['note']} (Category: {f['category'].title()})"
            )
        retrieved_context = "\n".join(context_parts)
        
        drug_names = ", ".join([d["label"] for d in drugs_added])
        
        prompt = f"""You are a helpful and compassionate clinical safety assistant.
The patient is taking the following medications: {drug_names}

We have retrieved the following verified interaction records from our clinical database:
{retrieved_context}

Please translate these technical medical records into a clear, patient-friendly summary in plain English.
Keep the response simple, short, and easy to read. Use bullet points for symptoms and actions.

For each flagged interaction, you MUST output exactly these three sections using bold headings:
1. **🚨 WHAT CAN GO WRONG**: Explain the danger in simple, non-technical terms.
2. **🤒 SYMPTOMS TO WATCH**: List the signs the patient should look out for (as short bullet points).
3. **✅ ACTION TO TAKE**: What the patient should do (as short bullet points, e.g. consult a doctor, separate timing).
"""
        explanation = None
        last_error = None
        for model_name in ['gemini-3.5-flash', 'gemini-3-flash-preview', 'gemini-2.5-flash']:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                explanation = response.text
                break
            except Exception as e:
                last_error = e
        
        if explanation:
            return explanation
        else:
            return f"❌ **Error calling Gemini API:** {str(last_error)}"
    except Exception as e:
        return f"❌ **Error calling Gemini API:** {str(e)}"


def generate_safe_drugs_explanation(drugs_added):
    """
    Call Gemini to generate a simple, patient-friendly explanation of each drug
    when no interactions are found.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass
            
    if not key:
        return "⚠️ **Gemini API Key missing.**"

    if not GENAI_AVAILABLE:
        return "❌ **GenAI SDK missing.**"

    try:
        client = genai.Client(api_key=key)
        
        drug_names = ", ".join([d["label"] for d in drugs_added])
        
        prompt = f"""You are a helpful and compassionate clinical safety assistant.
The patient is taking the following medications: {drug_names}
Our clinical database has confirmed that there are **no known drug-drug interactions** between these medications.

Please provide a very brief, patient-friendly overview of each drug to help them understand their treatment.
For each drug in the list, write:
1. **💊 What it is & Uses**: 1-2 simple sentences explaining what the drug is and what it is commonly used for.
2. **💡 Patient Care Tips**: 2-3 short, bulleted tips for safe usage (e.g., take with/without food, side effects to note).

Keep the explanations simple, short, and clean. Emphasize at the end that they are safe to take together according to current knowledge.
"""
        explanation = None
        last_error = None
        for model_name in ['gemini-3.5-flash', 'gemini-3-flash-preview', 'gemini-2.5-flash']:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                explanation = response.text
                break
            except Exception as e:
                last_error = e
        
        if explanation:
            return explanation
        else:
            return f"❌ **Error calling Gemini API:** {str(last_error)}"
    except Exception as e:
        return f"❌ **Error calling Gemini API:** {str(e)}"


# ── Session state init ────────────────────────────────────────────────────────

if "med_list" not in st.session_state:
    st.session_state.med_list = []   # list of {label, rxcuis: [(name, rxcui)]}

# ── Sidebar Config ────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown(
        """
        **System Mode:**
        * **RAG Prompting Engine**
        * Grounding: `app_data.db` (curated clinical safety pairs + openFDA labels)
        * LLM Model: `Gemini 3.5 Flash` (gemini-2.5-flash)
        """
    )
    if not GENAI_AVAILABLE:
        st.error("google-genai package not found. Run pip install google-genai")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("💊 India Drug Interaction Checker (RAG)")
st.caption(
    "⚠️ **Eval / Internal use only** — Grounded LLM Flagging Engine. "
    "No clinical sign-off. Not for patient-facing use without expert review."
)

col_left, col_right = st.columns([1, 1], gap="large")

# ── Left column: drug search & selection ─────────────────────────────────────

with col_left:
    st.subheader("Patient Medication List")

    query = st.text_input(
        "Search drug or brand name",
        placeholder="e.g.  Warfarin,  Augmentin,  Ecosprin 75 …",
        key="search_query",
    )

    if query:
        matches = search_drugs(query)
        if matches:
            options = {m["display"]: m for m in matches}
            chosen_display = st.selectbox(
                "Select drug",
                list(options.keys()),
                key="chosen",
                label_visibility="collapsed",
            )
            chosen = options[chosen_display]

            if st.button("➕  Add to list", use_container_width=True):
                # Avoid exact duplicates by label
                existing_labels = {d["label"] for d in st.session_state.med_list}
                if chosen["label"] not in existing_labels:
                    st.session_state.med_list.append(chosen)
                    st.rerun()
                else:
                    st.warning(f"{chosen['label']} is already in the list.")
        else:
            st.info("No matches found. Try a different spelling or generic name.")

    st.divider()

    # Current medication list
    if st.session_state.med_list:
        st.markdown(f"**{len(st.session_state.med_list)} drug(s) added:**")
        for i, drug in enumerate(st.session_state.med_list):
            c1, c2 = st.columns([5, 1])
            with c1:
                ing_str = ", ".join(f"{n}" for n, r in drug["rxcuis"][:5])
                prefix = "🏷️ " if drug.get("source") == "class" else ""
                st.markdown(f"**{prefix}{drug['label']}** — {ing_str}")
            with c2:
                if st.button("✕", key=f"remove_{i}"):
                    st.session_state.med_list.pop(i)
                    st.rerun()

        st.divider()

        if st.button("🔍  Check Interactions", type="primary", use_container_width=True):
            st.session_state.run_check = True

        if st.button("🗑️  Clear All", use_container_width=True):
            st.session_state.med_list = []
            st.session_state.pop("run_check", None)
            st.rerun()
    else:
        st.info("Add at least 2 drugs to check for interactions.")

# ── Right column: results ─────────────────────────────────────────────────────

with col_right:
    st.subheader("Interaction Results")

    if st.session_state.get("run_check") and len(st.session_state.med_list) >= 2:

        # Collect all RxCUIs from the list
        all_rxcuis = []
        for drug in st.session_state.med_list:
            for _, rxcui in drug["rxcuis"]:
                all_rxcuis.append(rxcui)

        flags = check_medlist(all_rxcuis)

        if not flags:
            st.success("✅  No known interactions found between these drugs.")
            
            # Call Gemini safe drugs overview Generator
            st.markdown("### 🤖 Medication Information (Gemini 3.5 Flash)")
            with st.spinner("Generating drug overviews..."):
                safe_explanation = generate_safe_drugs_explanation(
                    st.session_state.med_list
                )
                st.markdown(safe_explanation)
        else:
            st.error(f"⚠️  {len(flags)} interaction(s) flagged in database", icon="🚨")
            
            # Call Gemini RAG Generator
            st.markdown("### 🤖 Grounded Patient Guidance (Gemini 3.5 Flash)")
            with st.spinner("Generating patient-friendly summaries..."):
                explanation = generate_ddi_explanation(
                    st.session_state.med_list, flags
                )
                st.markdown(explanation)
            
            # Technical logs (collapsible)
            st.divider()
            with st.expander("🔬 Technical Database Details (For Clinicians)"):
                # Sort: critical first
                def sort_key(f):
                    cat = f["category"]
                    if cat in CRITICAL_CATS: return 0
                    if cat in HIGH_CATS:     return 1
                    return 2

                flags.sort(key=sort_key)
                for f in flags:
                    badge = severity_badge(f["category"])
                    color = severity_color(f["category"])
                    st.markdown(
                        f"""
                        <div style="
                            border-left: 4px solid {color};
                            padding: 10px 14px;
                            margin-bottom: 10px;
                            background: #1a1a1a;
                            border-radius: 4px;
                        ">
                            <strong>{badge}</strong> &nbsp; <code>{f['category']}</code>
                            <div style="font-size:1.05em; font-weight:600; margin: 4px 0;">
                                {f['name_a'].title()} &nbsp;+&nbsp; {f['name_b'].title()}
                            </div>
                            <div style="font-size:0.9em; color:#bbb;">
                                Mechanism: {f['note']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    elif st.session_state.get("run_check"):
        st.warning("Add at least 2 drugs first.")
    else:
        st.markdown(
            """
            **How to use:**
            1. Search for a drug or Indian brand name on the left
            2. Add it to the list
            3. Add more drugs (at least 2)
            4. Click **Check Interactions**
            
            *The app runs a local database search. If interactions are matched, Gemini 3.5 Flash generates grounded patient summaries.*
            """
        )

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
db_path = OUT_DIR / "app_data.db"
if db_path.exists():
    try:
        con = sqlite3.connect(str(db_path))
        total_pairs = con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        con.close()
    except Exception:
        total_pairs = 0
else:
    total_pairs = 0

st.caption(
    f"Unified Store: **{total_pairs:,} active DDI pairs** | "
    "Sources: Curated clinical pairs · OpenFDA drug labels · RxNorm crosswalk | "
    "RAG LLM Mode Activated"
)
