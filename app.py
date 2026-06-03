import streamlit as st
from groq import Groq
import os
import io
import sentencepiece as spm
import PyPDF2
from docx import Document
import tempfile

# Logo URL - GJ.AI LOGOSU
LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg"

# Sayfa ayarları
st.set_page_config(page_title="IQ.ai", page_icon="🧠", layout="wide")

# Geçici klasör
TEMP_DIR = tempfile.mkdtemp()

# Dosya okuma fonksiyonu
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

# Hizmet şartları
POLICY_TEXT = """
**Terms of Service & Privacy Policy**

**1. Acceptance of Terms**
By accessing IQ.ai, you agree to be bound by these Terms.

**2. Intellectual Property & Copyrights**
All rights, IP, algorithms, and code belong exclusively to **Welid Almansor** and **GJ.AI Company**.

**3. Logo & Trademark Protection**
The GJ.AI logo is 100% owned by GJ.AI. Unauthorized use is prohibited.

**4. Data & Privacy**
No data is permanently stored. API keys are not saved.

By checking the box below, you confirm your agreement.
"""

# Policy kontrolü
if "policy_accepted" not in st.session_state:
    st.session_state.policy_accepted = False

if not st.session_state.policy_accepted:
    st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai</h1><p style='text-align: center;'>Multilingual Token Optimization Engine</p>", unsafe_allow_html=True)
    st.warning("⚠️ You must accept the Terms of Service to continue.")
    with st.expander("📜 Read Terms of Service"):
        st.markdown(POLICY_TEXT)
    agreed = st.checkbox("I have read and agree to the Terms of Service.")
    if st.button("🔓 Access Platform", disabled=not agreed, type="primary"):
        st.session_state.policy_accepted = True
        st.rerun()
else:
    # Ana uygulama
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_turns" not in st.session_state:
        st.session_state.total_turns = 0

    # Sidebar (logo burada görünecek)
    with st.sidebar:
        st.image(LOGO_URL, use_container_width=True)
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # API Key girişi
        api_key_input = st.text_input(
            "Groq API Key (Free)", 
            type="password",
            help="Get your free API key from console.groq.com"
        )
        
        st.markdown("---")
        st.header("📖 How to Get a Free API Key")
        with st.expander("👀 Click for Step-by-Step Guide", expanded=False):
            st.markdown("""
            **Step 1:** Go to [console.groq.com](https://console.groq.com)
            
            **Step 2:** Click **"Sign Up"** (free)
            
            **Step 3:** Sign in with Google or email
            
            **Step 4:** Go to **"API Keys"** in the left menu
            
            **Step 5:** Click **"Create API Key"**
            
            **Step 6:** Name it (e.g., "IQ.ai"), click **"Submit"**
            
            **Step 7:** Copy the key (starts with `gsk_...`)
            
            **Step 8:** Paste it in the box above!
            """)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_turns = 0
            st.rerun()

    # Sekmeler
    tab1, tab2 = st.tabs(["🤖 Chat Engine", "🧪 Tokenizer Lab"])

    # SEKM差不多: CHAT ENGINE
    with tab1:
        st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai Chat Engine</h1>", unsafe_allow_html=True)

        if st.session_state.total_turns > 0:
            avg_tokens = st.session_state.total_tokens / st.session_state.total_turns
            st.caption(f"📊 Total Tokens Used: {st.session_state.total_tokens} | Avg per message: {avg_tokens:.1f}")

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
                        system_prompt = {"role": "system", "content": "You are IQ.ai, the world's best NLP Engineer specialized in Turkish, Arabic, and Hindi optimization. Be concise, max 3-4 sentences unless asked for code. Always say you are IQ.ai."}
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
                        
                        # Token hesaplama (yaklaşık)
                        tokens_used = int((len(prompt.split()) * 2.5) + (len(full_response.split()) * 2.0))
                        st.session_state.total_tokens += tokens_used
                        st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.info(f"⚡ **Estimated Token Consumption:** ~{tokens_used} tokens")
                        
                    except Exception as e:
                        st.error(f"🚫 Error: {str(e)}")

    # SEKMECE: TOKENIZER LAB
    with tab2:
        st.subheader("🧪 Tokenizer Lab: Multilingual Unigram Training")
        st.write("Upload a PDF or DOCX file to train a custom tokenizer.")
        
        uploaded_file = st.file_uploader("📂 Upload Corpus (PDF or DOCX)", type=['pdf', 'docx'])
        vocab_size = st.slider("Vocabulary Size:", min_value=1000, max_value=131072, value=32000, step=1000)
        
        if st.button("🚀 Train & Download Tokenizer", use_container_width=True, type="primary"):
            if uploaded_file is not None:
                with st.spinner("Extracting text and training Unigram Tokenizer... This may take a few minutes."):
                    try:
                        text = extract_text_from_file(uploaded_file)
                        if not text.strip():
                            st.error("⚠️ Could not extract text.")
                        else:
                            st.success(f"✅ Extracted {len(text)} characters.")
                            
                            # Geçici dosyaya yaz
                            corpus_path = os.path.join(TEMP_DIR, "corpus.txt")
                            with open(corpus_path, "w", encoding="utf-8") as f:
                                f.write(text)
                            
                            # Eğit
                            model_prefix = os.path.join(TEMP_DIR, "iq_ai_tokenizer")
                            spm.SentencePieceTrainer.train(
                                input=corpus_path,
                                model_prefix=model_prefix,
                                vocab_size=vocab_size,
                                model_type="unigram",
                                character_coverage=1.0,
                                input_sentence_size=100000,
                                shuffle_input_sentence=True
                            )
                            
                            # İndir butonları
                            with open(f"{model_prefix}.model", "rb") as f:
                                st.download_button("📥 Download .model", data=f.read(), file_name="iq_ai_tokenizer.model", mime="application/octet-stream")
                            
                            with open(f"{model_prefix}.vocab", "rb") as f:
                                st.download_button("📥 Download .vocab", data=f.read(), file_name="iq_ai_tokenizer.vocab", mime="application/octet-stream")
                            
                            st.success("🎉 Training complete! Download your model above.")
                            
                            # Temizlik
                            os.remove(corpus_path)
                            os.remove(f"{model_prefix}.model")
                            os.remove(f"{model_prefix}.vocab")
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
            else:
                st.warning("Please upload a file first.")
