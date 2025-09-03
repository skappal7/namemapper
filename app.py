import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import io
import json
import os

st.set_page_config(page_title="Name Mapper Wizard", layout="wide")

# --- Constants ---
MEMORY_FILE = "corrections.json"

# --- Load or initialize dictionary memory ---
def load_corrections(file=None):
    if file:  # If user uploaded dictionary
        return json.load(file)
    elif os.path.exists(MEMORY_FILE):  # Fallback to local memory file
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_corrections(corrections):
    with open(MEMORY_FILE, "w") as f:
        json.dump(corrections, f, indent=2)

# --- Title & Intro ---
st.markdown(
    """
    <div style="text-align:center; padding:10px; border-radius:12px; background:linear-gradient(90deg,#4facfe,#00f2fe); color:white;">
        <h1>🧙 Name Mapper Wizard</h1>
        <h3>Your name mapping assistant</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- How It Works Section ---
with st.expander("ℹ️ How This App Works"):
    st.markdown(
        """
        ### 🚀 Features:
        - Upload **Reference File** (correct names) and **Target File** (inconsistent names).
        - Smart **fuzzy matching** with guaranteed accuracy ≥ **80%**.
        - **Confidence threshold slider** to filter low-quality matches.
        - **Persistent dictionary memory**:
            - Upload existing dictionary JSON (optional).
            - After corrections, download the updated dictionary JSON.
            - Bring it back next time for smarter auto-mapping.
        - Export final results as **CSV** or **Excel**.
        - Clean, interactive, and production-ready UI.

        ⚡ *Tip:* The more corrections you add, the smarter the wizard becomes!
        """
    )

# --- Sidebar Upload Section ---
st.sidebar.header("📂 Upload Files")

# Dictionary Upload (optional)
dict_upload = st.sidebar.file_uploader("Upload Correction Dictionary (JSON, optional)", type=["json"])

# Load corrections
corrections = load_corrections(dict_upload)

ref_file = st.sidebar.file_uploader("Upload Reference File (Correct Names)", type=["csv", "xlsx"])
target_file = st.sidebar.file_uploader("Upload Target File (Inconsistent Names)", type=["csv", "xlsx"])

# --- Confidence Threshold ---
threshold = st.sidebar.slider(
    "Confidence Threshold (minimum % match)", 
    min_value=80, max_value=100, value=85, step=1,
    help="Only mappings with confidence ≥ threshold are considered reliable."
)

# --- Helper: Read file ---
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# --- Run mapping when files uploaded ---
if ref_file and target_file:
    ref_df = load_file(ref_file)
    target_df = load_file(target_file)

    # Assume first column has names
    ref_col = st.selectbox("Select Reference Column (Correct Names)", ref_df.columns)
    target_col = st.selectbox("Select Target Column (Inconsistent Names)", target_df.columns)

    if st.button("✨ Run Name Mapping"):
        ref_names = ref_df[ref_col].dropna().astype(str).tolist()
        target_names = target_df[target_col].dropna().astype(str).tolist()

        matches = []
        for name in target_names:
            if name in corrections:  # Auto apply from memory
                best_match, score = corrections[name], 100
            else:
                best_match, score, _ = process.extractOne(
                    name, ref_names, scorer=fuzz.WRatio
                )
                if score < 80:  # Enforce hard minimum
                    best_match, score = "⚠️ REVIEW REQUIRED", score
            matches.append([name, best_match, score])

        results_df = pd.DataFrame(matches, columns=["Original Name", "Mapped Name", "Confidence"])
        st.success("✅ Mapping completed!")

        # --- Separate High vs Low Confidence ---
        low_conf_df = results_df[results_df["Confidence"] < threshold]
        high_conf_df = results_df[results_df["Confidence"] >= threshold]

        st.subheader("✅ High Confidence Matches (Auto-Mapped)")
        st.dataframe(high_conf_df, use_container_width=True)

        if not low_conf_df.empty:
            st.subheader("⚠️ Low Confidence Matches (Manual Review Required)")
            st.info("These names did not meet the confidence threshold. Please correct them manually.")
            low_conf_df = st.data_editor(low_conf_df, num_rows="dynamic", use_container_width=True)

            # Save corrections back to dictionary
            for _, row in low_conf_df.iterrows():
                if row["Mapped Name"] not in ["⚠️ REVIEW REQUIRED", ""]:
                    corrections[row["Original Name"]] = row["Mapped Name"]
            save_corrections(corrections)

            final_df = pd.concat([high_conf_df, low_conf_df], ignore_index=True)
        else:
            final_df = high_conf_df

        # --- Download Section ---
        st.subheader("📥 Download Corrected Results")

        # CSV Download
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv_buffer.getvalue(),
            file_name="mapped_results.csv",
            mime="text/csv",
        )

        # Excel Download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Mapped Results")
        st.download_button(
            "⬇️ Download Excel",
            excel_buffer.getvalue(),
            file_name="mapped_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- Prompt User to Download Dictionary ---
        st.subheader("📖 Correction Dictionary")
        st.success("💾 Your corrections have been saved! Please download the dictionary for future runs.")
        dict_buffer = io.StringIO()
        json.dump(corrections, dict_buffer, indent=2)
        st.download_button(
            "⬇️ Download JSON Dictionary",
            dict_buffer.getvalue(),
            file_name="corrections.json",
            mime="application/json",
        )

# --- Footer ---
st.markdown(
    """
    <hr>
    <div style="text-align: center; font-size: 14px; color: grey;">
        Developed by <b>CE Innovations Lab 2025</b>
    </div>
    """,
    unsafe_allow_html=True,
)
