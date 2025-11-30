import streamlit as st
import pandas as pd

st.title("Upload Multiple Excel Files")

# File uploader for multiple Excel files
uploaded_files = st.file_uploader(
    "Drop or select Excel files here",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"**Filename:** {uploaded_file.name}")
        # Read Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        st.dataframe(df)
