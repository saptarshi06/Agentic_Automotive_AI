# app.py
import streamlit as st
import json, os, tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.makedirs("outputs/figures", exist_ok=True)

from extractor.azure_di import extract_layout, extract_with_figures
from extractor.versions import build_v1, build_v2, build_v3, build_v4
from extractor.ogx_rag import ingest_to_ogx, query_documents

st.set_page_config(page_title="Automotive Doc Extractor", layout="wide")
st.title("🚗 Automotive Document Extractor")
st.caption("Azure Document Intelligence (prebuilt-layout) + OGX v1.0")

uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, HTML, PNG, JPG)",
    accept_multiple_files=True,
    type=["pdf", "docx", "html", "png", "jpg", "jpeg"]
)

version = st.selectbox("Extraction Version", ["v1", "v2", "v3", "v4"])

col1, col2 = st.columns(2)
run_extract = col1.button("🔍 Extract JSON")
run_query = col2.button("💬 Ask across documents (RAG)")

if run_extract and uploaded_files:
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with st.spinner(f"Processing {file.name} via Azure DI..."):
            try:
                if version == "v1":
                    result, figures = extract_with_figures(tmp_path)
                    output = build_v1(result, figures, file.name)
                elif version == "v2":
                    result, figures = extract_with_figures(tmp_path)
                    output = build_v2(result, file.name)
                elif version == "v3":
                    result = extract_layout(tmp_path)
                    output = build_v3(result, file.name)
                else:  # v4
                    result = extract_layout(tmp_path)
                    output = build_v4(result, file.name)

                st.subheader(f"📄 {file.name} — {version}")
                st.json(output)
                st.download_button(
                    f"⬇ Download {file.name}_{version}.json",
                    data=json.dumps(output, indent=2),
                    file_name=f"{file.name}_{version}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        os.unlink(tmp_path)

if run_query and uploaded_files:
    question = st.text_input("Ask a question across all uploaded documents:")
    if question:
        vs_ids = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                tmp.write(file.read())
                vs_id = ingest_to_ogx(tmp.name)
                vs_ids.append(vs_id)
                os.unlink(tmp.name)

        with st.spinner("OGX searching documents..."):
            answer = query_documents(vs_ids, question)
            st.write(answer)