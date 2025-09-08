import streamlit as st
import requests

st.title("MedRAG Assistant")

query = st.text_input("Ask about a drug:")
if st.button("Submit"):
    r = requests.post("http://localhost:8000/chat", json={"q": query})
    data = r.json()
    st.markdown(data["answer"])
    st.write("References:")
    for ref in data["references"]:
        st.write(ref)