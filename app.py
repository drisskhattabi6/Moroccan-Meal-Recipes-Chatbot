import re, io, os
import subprocess
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from rag import RAGSystem

def generate_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 40  # Start position for text
    max_width = width - 80  # Margin for text wrapping

    c.setFont("Helvetica-Bold", 16)
    header_text = "Conversation History"
    text_width = c.stringWidth(header_text, "Helvetica-Bold", 16)
    c.drawString((width - text_width) / 2, height - 40, header_text)

    y -= 30  # Adjust position after header
    c.setFont("Helvetica", 12)

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "LLM"
        text = f"{role}: {msg['content']}"

        # Separate user questions with a line
        if msg["role"] == "user":
            y -= 10
            c.setStrokeColorRGB(0, 0, 0)
            c.line(40, y, width - 40, y)
            y -= 20  

        # Wrap text within max_width
        wrapped_lines = simpleSplit(text, c._fontname, c._fontsize, max_width)

        for line in wrapped_lines:
            c.drawString(40, y, line)
            y -= 20
            
            # Handle page breaks
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - 40  # Reset position after new page

    c.save()
    buffer.seek(0)
    return buffer

def get_available_models():
    """Fetches the installed Ollama models, excluding 'NAME' and models containing 'embed'."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = [
            line.split(" ")[0] for line in result.stdout.strip().split("\n")
            if line and "NAME" not in line and "embed" not in line.lower()
        ]
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error fetching models: {e}")
        return []

def format_time(response_time):
    minutes = response_time // 60
    seconds = response_time % 60
    return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"

def remove_tags(text):
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

# Fetch available models
available_models = get_available_models()
if not available_models:
    st.error("No installed Ollama models found. Please install one using `ollama pull <model_name>`.")

# Streamlit page configuration
st.set_page_config(page_title="Moroccan Meals & Recipes Chatbot", page_icon="🤖")
st.markdown("#### 🗨️ Moroccan Meals & Recipes Chatbot")

with st.sidebar:
    st.title("Settings :")

    # User selects the model Provider
    llm_provider = st.selectbox("Select LLM Provider:", ['Ollama', 'Openrouter'], index=0)

    if llm_provider == 'Ollama' :
        selected_model = st.selectbox("Select an Ollama model:", available_models, index=0)
    else : 
        llm_name = st.text_input("Enter LLM Name", value='qwen/qwq-32b:free')
        openrouter_api_key = st.text_input("Enter Openrouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY"))

    # Slider to choose the number of retrieved results
    n_results = st.slider("Number of retrieved documents", min_value=1, max_value=15, value=5)

    # Button to download PDF
    if st.button("Download Chat as PDF"):
        pdf_buffer = generate_pdf()
        st.download_button(
            label="Download",
            data=pdf_buffer,
            file_name="chat_history.pdf",
            mime="application/pdf"
        )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the RAG system
rag_system = RAGSystem(collection_name="moroccan_recipes", db_path="Moroccan_Recipes_FaissDB", n_results=n_results)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask Me..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                if llm_provider == 'Ollama' :
                    response_placeholder = st.empty()
                    streamed_response = ""

                    for chunk in rag_system.generate_response(query, selected_model):  # Stream response
                        if isinstance(chunk, dict):
                            metadata = chunk
                        else:
                            streamed_response = chunk 
                            response_placeholder.markdown(streamed_response)

                    st.write(f"""\n\n----
                    LLM Name : {selected_model} | Token Count: {metadata.get('token_count', 'N/A')} | Response Time: {metadata.get('response_time', 'N/A')} | n_results of context: {n_results}""")

                    response = f"""
                    {remove_tags(streamed_response)}
                    \n\n----
                    LLM Name : {selected_model} | Total Tokens: {metadata.get('token_count', 'N/A')} | Response Time: {metadata.get('response_time', 'N/A')} | n_results of context: {n_results}
                    """
                else :
                    llm_response, total_tokens = rag_system.generate_response2(query=query, llm_name=llm_name, openrouter_api_key=openrouter_api_key)
                    response = f"""
                    {llm_response}
                    \n----
                    LLM Name : {llm_name} | Total Tokens : {total_tokens} | n_results of context: {n_results}
                    """
                    st.write(response)

            # Display and store assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()
