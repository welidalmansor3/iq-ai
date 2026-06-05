import streamlit as st
from groq import Groq
import os
import sentencepiece as spm

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="IQ.ai | Token Optimizer", page_icon="🧠", layout="wide")

# --- KARANLIK TEMA (DARK MODE) CSS ---
st.markdown("""
<style>
    /* Ana Arka Plan */
    [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #050509 !important; color: #e0e0e0 !important; font-family: 'Inter', sans-serif;
    }
    /* Sol Menü Arka Planı */
    [data-testid="stSidebar"] { background-color: #0a0a0f !important; border-right: 1px solid #252530; }
    
    /* Başlıklar */
    h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
    
    /* Butonlar (Mavi-Mor Gradient) */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white; border: none; padding: 12px 24px; font-weight: bold;
        border-radius: 10px; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(37, 117, 252, 0.3); }
    
    /* Kart Tasarımı */
    .card {
        background: linear-gradient(145deg, #101014, #16161a); padding: 25px; border-radius: 14px; 
        border: 1px solid #252530; box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin-bottom: 15px;
    }
    
    /* Metin Girişleri */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #101014 !important; color: #e0e0e0 !important; border-color: #252530 !important;
    }
    
    /* Linkler ve Alt Metin */
    a { color: #2575fc !important; } .subtext { color: #a0a0a0; }
</style>
""", unsafe_allow_html=True)

# LOGO URL
LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg?auth_key=1880000279-e3e53963895d4cb2b17766ad29dd2480-0-3f2ced5648a41f4923250c661dc275fd"

# MODEL YOLLARI
DEFAULT_MODEL = "iq_ai_tokenizer.model"

POLICY_TEXT = """
**Terms of Service & Privacy Policy**

**1. Acceptance of Terms**
By accessing IQ.ai ("the Platform"), you agree to be bound by these Terms.

**2. Intellectual Property & Copyrights**
All rights, IP, algorithms, and code belong exclusively to **Welid Almansor / GJ.AI Company**.

**3. Logo & Trademark Protection**
The GJ.AI logo is 100% owned. Unauthorized use is strictly prohibited and subject to lawsuit.

**4. Data Privacy**
Chat inputs are processed securely. Trained models are hosted for demonstration.

By checking the box below, you confirm your agreement.
"""

if "policy_accepted" not in st.session_state: st.session_state.policy_accepted = False

if not st.session_state.policy_accepted:
    st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai</h1><p style='text-align: center; color: #a0a0a0;'>Multilingual Token Optimization Engine</p><hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)
    st.warning("⚠️ **Authorization Required:** You must accept the Terms of Service.")
    with st.expander("📜 READ: Terms of Service & Privacy Policy", expanded=True): st.markdown(POLICY_TEXT)
    agreed = st.checkbox("I have read and I agree to the Terms of Service and Privacy Policy.")
    if st.button("🔓 Access Platform", disabled=not agreed, type="primary", use_container_width=True):
        st.session_state.policy_accepted = True; st.rerun()

else:
    if "messages" not in st.session_state: st.session_state.messages = []
    if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
    if "total_turns" not in st.session_state: st.session_state.total_turns = 0

    with st.sidebar:
        st.image(LOGO_URL, use_container_width=True); st.markdown("---")
        st.header("⚙️ Settings"); api_key_input = st.text_input("Groq API Key", type="password")
        with st.expander("📖 Get Free API Key"): st.markdown("1. Go to [Groq Console](https://console.groq.com)\n2. Sign Up -> API Keys -> Create\n3. Copy `gsk_...`")
        st.markdown("---")
        if st.button("🗑️ Clear Chat"): st.session_state.messages = []; st.session_state.total_tokens = 0; st.session_state.total_turns = 0; st.rerun()

    tab1, tab2, tab3 = st.tabs(["🤖 Chat Engine", "📊 Token Comparison", "🛠️ Surgery Scripts"])

    # ==========================================
    # TAB 1: CHAT ENGINE
    # ==========================================
    with tab1:
        st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai Chat Engine</h1><p class='subtext' style='text-align: center;'>Short, precise answers. Token-optimized.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        if st.session_state.total_turns > 0: st.caption(f"📊 Total Tokens: {st.session_state.total_tokens} | Avg: {st.session_state.total_turns:.1f}")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("Type your message here..."):
            if not api_key_input: st.error("🚫 Enter Groq API key in sidebar.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    try:
                        client = Groq(api_key=api_key_input)
                        sys_pr = {"role": "system", "content": "You are IQ.ai, the world's best NLP Engineer. Be concise."}
                        stream = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[sys_pr] + st.session_state.messages, temperature=0.2, max_tokens=4000, stream=True)
                        full_response = "".join([chunk.choices[0].delta.content or "" for chunk in stream])
                        st.markdown(full_response)
                        tokens_used = int((len(prompt.split()) * 2.5) + (len(full_response.split()) * 2.0))
                        st.session_state.total_tokens += tokens_used; st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e: st.error(f"🚫 Error: {str(e)}")

    # ==========================================
    # TAB 2: TOKEN COMPARISON (GPT-4 vs IQ.ai)
    # ==========================================
    with tab2:
        st.markdown("<h1 style='text-align: center;'>📊 Live Token Tax Benchmark</h1><p class='subtext' style='text-align: center;'>GPT-4 Tokenizer vs IQ.ai Custom Tokenizer.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        
        if os.path.exists(DEFAULT_MODEL):
            sp = spm.SentencePieceProcessor(); sp.load(DEFAULT_MODEL)
            st.success("✅ IQ.ai Custom Multilingual Tokenizer is Active!")
            
            # GPT-4 Tokenizer'ı yükle
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-4")
                gpt_active = True
            except:
                gpt_active = False
                st.warning("GPT-4 tokenizer could not be loaded. Comparison will be estimated based on industry averages.")
            
            test_sentences = {
                "🇹🇷 Turkish": "İş arayanları sahte ve hayalet ilanlardan koruyoruz.",
                "🇸🇦 Arabic": "نحن نحمي الباحثين عن عمل من الإعلانات المزيفة.",
                "🇮🇳 Hindi": "हम नौकरी तलाशने वालों को नकली विज्ञापनों से बचाते हैं।"
            }
            
            for lang, sentence in test_sentences.items():
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{lang}**")
                st.markdown(f"<p class='subtext'><i>\"{sentence}\"</i></p>", unsafe_allow_html=True)
                
                # GPT-4 Token Sayısı
                if gpt_active:
                    gpt_tokens = enc.encode(sentence)
                    gpt_count = len(gpt_tokens)
                else:
                    gpt_count = int(len(sentence.split()) * 2.5) 
                
                # IQ.ai Token Sayısı
                iq_tokens = sp.encode(sentence, out_type=str)
                iq_count = len(iq_tokens)
                
                # Gerçek Tasarruf Yüzdesi
                reduction = 100 - ((iq_count / gpt_count) * 100) if gpt_count > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1: 
                    st.metric("GPT-4 Tokenizer", f"{gpt_count} Tokens")
                with col2: 
                    st.metric("IQ.ai Unigram", f"{iq_count} Tokens")
                with col3: 
                    st.metric("Token Reduction", f"%{reduction:.1f}", delta="Saved", delta_color="normal")
                
                # Kullanıcıya tokenların nasıl parçalandığını göster
                with st.expander("🔍 GPT-4 vs IQ.ai Token Breakdown"):
                    if gpt_active:
                        st.write("**GPT-4 Fragments:**", [enc.decode([t]) for t in gpt_tokens])
                    st.write("**IQ.ai Fragments:**", iq_tokens)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Trained model (`iq_ai_tokenizer.model`) not found in repository. Please ensure it is uploaded to GitHub.")

    # ==========================================
    # TAB 3: SURGERY SCRIPTS
    # ==========================================
    with tab3:
        st.markdown("<h1 style='text-align: center;'>🛠️ Surgery Scripts</h1><p class='subtext' style='text-align: center;'>For Enterprise Integration.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        with st.expander("🩺 Step 1: Mean-Composition Surgery"):
            st.code("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nmodel = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')\nold_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')\n# Load your trained IQ.ai tokenizer and add tokens...\n# model.resize_token_embeddings(len(old_tokenizer))", language='python')
        with st.expander("📦 Step 2: 4-Bit Quantization"):
            st.code("from awq import AutoAWQForCausalLM\nmodel = AutoAWQForCausalLM.from_pretrained('model_path')\nmodel.quantize(tokenizer, quant_config={ 'zero_point': True, 'q_group_size': 128, 'w_bit': 4 })", language='python')
