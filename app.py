import streamlit as st
from groq import Groq
import os
import io
import sentencepiece as spm
import PyPDF2
from docx import Document
import tempfile

# Sayfa ayarları
st.set_page_config(page_title="IQ.ai", page_icon="🧠", layout="wide")

# Logo URL (GJ.AI logosu)
LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg"

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

    # Sidebar
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

    # SEKM
