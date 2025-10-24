import streamlit as st
import requests

# Set the title of the web application
st.title("MedRAG Assistant")

# Create input text field for user query
query = st.text_input("Ask about a drug:")

# Create submit button and handle click event
if st.button("Submit"):
    # Send POST request to FastAPI backend
    r = requests.post("http://localhost:8000/chat", json={"q": query})
    
    # Parse JSON response
    data = r.json()
    
    # Display the AI generated answer
    st.markdown(data["answer"])
    
    # Display reference links section
    st.write("References:")
    for ref in data["references"]:
        st.write(ref)