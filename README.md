# India Drug-Drug Interaction Checker (RAG-Enabled)

An intelligent, data-grounded clinical safety tool designed to flag pairwise drug-drug interactions (DDI) for commercial Indian drug products. It resolves brand names to standard active generic ingredients, checks a local SQLite grounding store, and utilizes a multi-model Gemini API fallback chain to generate patient-friendly warnings.

---

## 🚀 Key Features

* **Brand-to-Ingredient Resolution**: Resolves brand names (e.g. *Augmentin*) to their active generic ingredients (e.g. *Amoxycillin* and *Clavulanic Acid*) to prevent FDC brand-masking errors.
* **Synonym Normalization**: Maps drug names and synonyms (e.g. *Paracetamol* and *Acetaminophen*) to standard RxNorm RxCUIs to prevent spelling/naming misses.
* **Consolidated SQLite Grounding Store**: Queries `data/out/app_data.db` (containing **2,596 active interaction rules** mined from curated clinical safety seeds and openFDA labels) in under 0.05ms.
* **Grounded LLM Generation (RAG)**: If an interaction is flagged in the database, the mechanism notes are passed to the Gemini API to write a simple, patient-friendly summary.
* **Safe Medication Overviews**: If no interactions are found, the LLM provides patient overviews and 2-3 standard patient safety tips for each added drug.
* **Multi-Model Fallback Chain**: Automatic try-catch loop trying Gemini 3.5 Flash (`gemini-3.5-flash`), falling back to Gemini 3 Flash (`gemini-3-flash-preview`), and Gemini 2.5 Flash (`gemini-2.5-flash`).

---

## 📁 Repository Structure (Pruned for Cloud)

For cloud deployment, the workspace has been pruned to exclude heavy build-time files (reducing project payload size from **~1.7 GB to 195 MB**):

```
medicine/
├── app.py                 # Streamlit web application entrypoint
├── requirements.txt       # Python package dependencies
├── .env                   # Environment variable template file
└── data/
    └── out/
        └── app_data.db    # Consolidated SQLite database (194.5 MB)
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed. Install the package dependencies:
```bash
pip install -r requirements.txt
```

*Requirements content:*
* `streamlit`
* `pandas`
* `google-genai`
* `python-dotenv` (optional)

### 2. Configure the Gemini API Key
Create a `.env` file in the root directory (or use the provided template) and add your API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
*(On cloud hosting platforms like Streamlit Community Cloud or Render, configure `GEMINI_API_KEY` directly inside the platform's Environment Variables/Secrets settings).*

---

## 🏃 Running the Application

Launch the Streamlit web application on your local machine:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔬 Clinical Disclaimer
**Eval / Internal Use Only**: This application is a first-pass interaction flagger for internal evaluation only. It contains no clinical sign-off and is **not** to be used for clinical decision support or direct patient-facing use without expert medical review.
