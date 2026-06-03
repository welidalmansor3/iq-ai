import streamlit as st
from groq import Groq
import os
import io
import sentencepiece as spm
import PyPDF2
from docx import Document

# LOGO URL
LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg?auth_key=1880000279-e3e53963895d4cb2b17766ad29dd2480-0-3f2ced5648a41f4923250c661dc275fd"

# GEÇİCİ KLASÖR
import tempfile
TEMP_DIR = tempfile.mkdtemp()
MODEL_PREFIX = os.path.join(TEMP_DIR, "iq_ai_tokenizer")

def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# POLICY
POLICY_TEXT = """
**Terms of Service & Privacy Policy**

**1. Acceptance of Terms**
By accessing IQ.ai, you agree to be bound by these Terms.

**2. Intellectual Property & Copyrights**
All rights belong exclusively to Welid Almansor and GJ.AI Company.

**3. Data & Privacy**
No data is permanently stored.

By checking the box below, you confirm your agreement.
"""

if "policy_accepted" not in st.session_state:
    st.session_state.policy_accepted = False

if not st.session_state.policy_accepted:
    st.set_page_config(page_title="IQ.ai", page_icon="🧠", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai</h1><p style='text-align: center;'>Multilingual Token Optimization Engine</p>", unsafe_allow_html=True)
    st.warning("⚠️ You must accept the Terms of Service.")
    with st.expander("📜 READ: Terms of Service"):
        st.markdown(POLICY_TEXT)
    agreed = st.checkbox("I have read and agree to the Terms.")
    if st.button("🔓 Access Platform", disabled=not agreed, type="primary"):
        st.session_state.policy_accepted = True
        st.rerun()
else:
    st.set_page_config(page_title="IQ.ai", page_icon="🧠", layout="wide")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_turns" not in st.session_state:
        st.session_state.total_turns = 0

    with st.sidebar:
        st.image(LOGO_URL, use_container_width=True)
        st.markdown("---")
        st.header("⚙️ Settings")
        api_key_input = st.text_input("Groq API Key", type="password")
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_turns = 0
            st.rerun()

    tab1, tab2 = st.tabs(["🤖 Chat Engine", "🧪 Tokenizer Lab"])

    # TAB 1: CHAT ENGINE
    with tab1:
        st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai Chat Engine</h1>", unsafe_allow_html=True)

        if st.session_state.total_turns > 0:
            avg_tokens = st.session_state.total_tokens / st.session_state.total_turns
            st.caption(f"📊 Total Tokens: {st.session_state.total_tokens} | Avg: {avg_tokens:.1f}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            if not api_key_input:
                st.error("🚫 Please enter your Groq API key in the sidebar.")
            else:
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        client = Groq(api_key=api_key_input)
                        system_prompt = {"role": "system", "content": "You are IQ.ai, an NLP engineer specializing in Turkish, Arabic, and Hindi optimization. Be concise."}
                        api_messages = [system_prompt] + st.session_state.messages
                        
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=api_messages,
                            temperature=0.2,
                            max_tokens=4000,
                            stream=True
                        )
                        
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        
                        tokens_used = int((len(prompt.split()) * 2.5) + (len(full_response.split()) * 2.0))
                        st.session_state.total_tokens += tokens_used
                        st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.info(f"⚡ Estimated tokens: ~{tokens_used}")
                        
                    except Exception as e:
                        st.error(f"🚫 Error: {str(e)}")

    # TAB 2: TOKENIZER LAB
    with tab2:
        st.subheader("🧪 Tokenizer Lab")
        st.write("Upload a PDF or DOCX to train a custom tokenizer.")
        
        uploaded_file = st.file_uploader("📂 Upload file", type=['pdf', 'docx'])
        
        if st.button("🚀 Train Tokenizer", type="primary"):
            if uploaded_file is not None:
                with st.spinner("Extracting text and training..."):
                    text = extract_text_from_file(uploaded_file)
                    if text.strip():
                        st.success(f"✅ Extracted {len(text)} characters.")
                        
                        # Corpus'u geçici dosyaya yaz
                        corpus_path = os.path.join(TEMP_DIR, "corpus.txt")
                        with open(corpus_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        
                        # Eğit
                        spm.SentencePieceTrainer.train(
                            input=corpus_path,
                            model_prefix=MODEL_PREFIX,
                            vocab_size=32000,
                            model_type="unigram",
                            character_coverage=1.0
                        )
                        
                        # İndir butonu
                        with open(f"{MODEL_PREFIX}.model", "rb") as f:
                            st.download_button("📥 Download .model", data=f.read(), file_name="iq_ai_tokenizer.model")
                        
                        with open(f"{MODEL_PREFIX}.vocab", "rb") as f:
                            st.download_button("📥 Download .vocab", data=f.read(), file_name="iq_ai_tokenizer.vocab")
                        
                        st.success("✅ Training complete! Download your model.")
                    else:
                        st.error("Could not extract text.")
            else:
                st.warning("Please upload a file first.")