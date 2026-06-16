import streamlit as st
from groq import Groq
import os
import sentencepiece as spm
import tiktoken
import re
import time
import json
import tempfile
import pandas as pd
import plotly.express as px
from duckduckgo_search import DDGS
from transformers import AutoTokenizer

st.set_page_config(page_title="IQ.ai | Token Optimizer", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"], .main, .block-container { background-color: #050509 !important; color: #e0e0e0 !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0a0a0f !important; border-right: 1px solid #252530; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
    .stButton>button { background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); color: white; border: none; padding: 12px 24px; font-weight: bold; border-radius: 10px; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(37, 117, 252, 0.3); }
    .card { background: linear-gradient(145deg, #101014, #16161a); padding: 25px; border-radius: 14px; border: 1px solid #252530; box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin-bottom: 15px; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: #101014 !important; color: #e0e0e0 !important; border-color: #252530 !important; }
    a { color: #2575fc !important; } .subtext { color: #a0a0a0; }
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://z-cdn-media.chatglm.cn/files/c932e5f3-aa8b-4fb9-a0d0-fd57c9056545.jpeg"
DEFAULT_MODEL = "iq_ai_tokenizer.model"

# --- TOKENIZER YÜKLEMELERİ ---
@st.cache_resource
def load_base_tokenizers():
    gpt4 = tiktoken.encoding_for_model("gpt-4")
    llama = AutoTokenizer.from_pretrained("unsloth/llama-3-8b")
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    return gpt4, llama, qwen

gpt4_enc, llama_tok, qwen_tok = load_base_tokenizers()

# Chat Engine (Tab 1) için varsayılan model
sp_iq = None
if os.path.exists(DEFAULT_MODEL):
    sp_iq = spm.SentencePieceProcessor()
    sp_iq.load(DEFAULT_MODEL)

# --- YARDIMCI FONKSİYONLAR ---
def search_web(query, max_results=3):
    try:
        with DDGS() as ddgs:
            return [{"title": r.get("title",""), "url": r.get("href",""), "body": r.get("body","")} for r in ddgs.text(query, max_results=max_results)]
    except: return []

def run_integrity_check(client, question, answer):
    web_context = search_web(question)
    context_str = json.dumps(web_context, indent=2)
    prompt = f"Question: {question}\nModel Answer: {answer}\nWeb Evidence: {context_str}\nCalculate: 1. source_match (0-100) 2. consistency (0-100) 3. claim_verification (0-100) 4. hallucination_score (0-100). Return ONLY valid JSON: {{'source_match': int, 'consistency': int, 'claim_verification': int, 'hallucination_score': int, 'reason': 'short reason'}}"
    try:
        result = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0)
        text = re.sub(r'```json|```', '', result.choices[0].message.content.strip())
        data = json.loads(text)
        source_match = data.get("source_match", 50); consistency = data.get("consistency", 50); claim_ver = data.get("claim_verification", 50); hall_score = data.get("hallucination_score", 50)
        confidence = int((source_match * 0.4) + (consistency * 0.3) + (claim_ver * 0.3))
        return {"confidence": min(100, max(0, confidence)), "source_match": source_match, "consistency": consistency, "claim_verification": claim_ver, "hallucination_score": hall_score, "risk": "HIGH" if hall_score > 60 else "MEDIUM" if hall_score > 30 else "LOW", "reason": data.get("reason", "N/A"), "sources": web_context}
    except Exception as e:
        return {"confidence": 50, "source_match": 50, "consistency": 50, "claim_verification": 50, "hallucination_score": 50, "risk": "PARSE_ERR", "reason": str(e), "sources": []}

def generate_health_report(token_reduction, confidence, hallucination_score, latency):
    token_score = min(100, token_reduction * 2) if token_reduction > 0 else 0; cost_score = token_score; safety_score = 100 - hallucination_score; latency_score = max(0, 100 - (latency / 50))
    overall = int((confidence * 0.30) + (safety_score * 0.30) + (token_score * 0.15) + (cost_score * 0.15) + (latency_score * 0.10))
    return {"Token Efficiency": token_score, "Cost Score": cost_score, "Safety Score": safety_score, "Latency Score": latency_score, "Confidence": confidence, "Overall": overall}

BENCHMARK_SET = [
    {"category": "General", "question": "Türkiye'nin başkenti nedir?", "answer": "Ankara"}, {"category": "General", "question": "Suyun kimyasal formülü nedir?", "answer": "H2O"},
    {"category": "Math", "question": "2+2 kaç eder?", "answer": "4"}, {"category": "Math", "question": "10 * 5 kaçtır?", "answer": "50"},
    {"category": "Logic", "question": "Tüm kediler hayvansa, kediler nefes alır mı?", "answer": "Evet"},
    {"category": "Multilingual", "question": "How do you say 'Hello' in Spanish?", "answer": "Hola"},
]

def run_benchmark(client, benchmark_set):
    categories = {}; correct_total = 0
    for item in benchmark_set:
        try:
            result = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":item["question"]}], temperature=0)
            answer = result.choices[0].message.content; cat = item["category"]
            if cat not in categories: categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if item["answer"].lower() in answer.lower(): categories[cat]["correct"] += 1; correct_total += 1
        except: pass
    accuracy = round(correct_total/len(benchmark_set)*100, 2) if benchmark_set else 0
    return accuracy, categories

POLICY_TEXT = "**Terms of Service & Privacy Policy**\n**1. Acceptance of Terms** By accessing IQ.ai, you agree to be bound by these Terms.\n**2. Intellectual Property** All rights, IP, algorithms, and code belong exclusively to **Welid Almansor / GJ.AI Company**.\n**3. Logo & Trademark Protection** The GJ.AI logo is 100% owned. Unauthorized use is strictly prohibited.\n**4. Data Privacy** Chat inputs are processed securely."

if "policy_accepted" not in st.session_state: st.session_state.policy_accepted = False
if "sp_active" not in st.session_state: st.session_state.sp_active = None

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

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Chat Engine", "📊 Token Comparison", "🛠️ Surgery Scripts", "🎯 Benchmark Center"])

    # ==========================================
    # TAB 1: CHAT ENGINE
    # ==========================================
    with tab1:
        st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai Chat Engine</h1><p class='subtext' style='text-align: center;'>Short, precise answers. Token-optimized.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        if st.session_state.total_turns > 0: st.caption(f"📊 Total Tokens: {st.session_state.total_tokens} | Avg: {st.session_state.total_tokens/st.session_state.total_turns:.1f}")
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
                        
                        start_time = time.time()
                        stream = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[sys_pr] + st.session_state.messages, temperature=0.2, max_tokens=4000, stream=True)
                        full_response = "".join([chunk.choices[0].delta.content or "" for chunk in stream])
                        end_time = time.time()
                        latency_ms = int((end_time - start_time) * 1000)
                        st.markdown(full_response)
                        
                        # Chat engine için yüklü modeli kullan, yoksa default IQ modeli
                        sp_chat = st.session_state.sp_active if st.session_state.sp_active else sp_iq
                        
                        gpt_total = len(gpt4_enc.encode(prompt)) + len(gpt4_enc.encode(full_response))
                        iq_total = len(sp_chat.encode(prompt)) + len(sp_chat.encode(full_response)) if sp_chat else 0
                        token_reduction = round(100 - ((iq_total / gpt_total) * 100), 2) if (sp_chat and gpt_total > 0) else 0
                        
                        st.session_state.total_tokens += gpt_total; st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        with st.spinner("🔍 Running Integrity Check..."):
                            integrity = run_integrity_check(client, prompt, full_response)
                            report = generate_health_report(token_reduction=token_reduction, confidence=integrity["confidence"], hallucination_score=integrity["hallucination_score"], latency=latency_ms)
                            
                            st.markdown("---")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("🛡 Confidence", f"{integrity['confidence']}%")
                            with col2: st.metric("🚨 Hall. Risk", integrity['risk'])
                            with col3: st.metric("💰 Token Saved", f"%{token_reduction}")
                            with col4: st.metric("🏆 Overall Health", f"{report['Overall']}")

                            with st.expander("🔬 Detailed AI Integrity Analysis"):
                                sc1, sc2, sc3 = st.columns(3)
                                with sc1: st.metric("Source Match", f"{integrity['source_match']}%")
                                with sc2: st.metric("Consistency", f"{integrity['consistency']}%")
                                with sc3: st.metric("Claim Verification", f"{integrity['claim_verification']}%")
                                st.json({"Risk Level": integrity['risk'], "Hallucination Score": f"{integrity['hallucination_score']}/100", "Reason": integrity['reason']})
                                if integrity['sources']:
                                    for src in integrity['sources']: st.markdown(f"**{src.get('title', '')}** - [Link]({src.get('url', '#')})")
                    except Exception as e: st.error(f"🚫 Error: {str(e)}")

    # ==========================================
    # TAB 2: TOKEN COMPARISON (DOSYA YÜKLEME + GRAFİK + ROI)
    # ==========================================
    with tab2:
        st.markdown("<h1 style='text-align: center;'>📊 Live Token Tax Benchmark</h1><p class='subtext' style='text-align: center;'>GPT-4 vs Llama 3 vs Qwen 2.5 vs Your Custom Model</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        
        st.subheader("📂 Load Your Tokenizer Model")
        st.markdown("Select your `.model` file below to start the comparison.")
        
        uploaded_file = st.file_uploader("Choose a .model file", type=["model"])
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".model") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                sp_custom = spm.SentencePieceProcessor()
                sp_custom.load(tmp_file_path)
                st.success("✅ Model Loaded Successfully! Comparing now...")
                st.session_state.sp_active = sp_custom # Yüklenen modeli oturuma kaydet
            except Exception as e: 
                st.error(f"🚫 Error loading model: {e}")
                st.session_state.sp_active = None
        
        sp_tab2 = st.session_state.sp_active
        
        if sp_tab2:
            test_sentences = {
                "🇹🇷 Turkish": "İş arayanları sahte ve hayalet ilanlardan koruyoruz.", 
                "🇸🇦 Arabic": "نحن نحمي الباحثين عن عمل من الإعلانات المزيفة.", 
                "🇮🇳 Hindi": "हम नौकरी तलाशने वालों को नकली विज्ञापनों से बचाते हैं।"
            }
            
            total_reduction = 0
            lang_count = 0
            
            for lang, sentence in test_sentences.items():
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{lang}**")
                st.markdown(f"<p class='subtext'><i>\"{sentence}\"</i></p>", unsafe_allow_html=True)
                
                gpt_count = len(gpt4_enc.encode(sentence))
                llama_count = len(llama_tok.encode(sentence, add_special_tokens=False))
                qwen_count = len(qwen_tok.encode(sentence, add_special_tokens=False))
                custom_count = len(sp_tab2.encode(sentence)) 
                reduction = 100 - ((custom_count / gpt_count) * 100) if gpt_count > 0 else 0
                
                total_reduction += reduction
                lang_count += 1
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("🔵 GPT-4", f"{gpt_count} Tok")
                with col2: st.metric("🦙 Llama 3", f"{llama_count} Tok")
                with col3: st.metric("🟢 Qwen 2.5", f"{qwen_count} Tok")
                with col4: st.metric("🧠 Your Model", f"{custom_count} Tok")
                with col5: st.metric("📉 Reduction", f"%{reduction:.1f}", delta="Saved", delta_color="normal")
                
                # ==========================================
                # 1. EKLENEN: ÇUBUK GRAFİK (BAR CHART)
                # ==========================================
                df = pd.DataFrame({
                    "Model": ["GPT-4", "Llama 3", "Qwen 2.5", "Your Model"],
                    "Token Count": [gpt_count, llama_count, qwen_count, custom_count]
                })
                
                fig = px.bar(df, x="Model", y="Token Count", color="Model", text="Token Count",
                             color_discrete_map={"GPT-4": "#6a11cb", "Llama 3": "#2575fc", "Qwen 2.5": "#00d2ff", "Your Model": "#00f260"})
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='white',
                    title=f'{lang} Token Consumption',
                    xaxis_title="Models",
                    yaxis_title="Tokens",
                    showlegend=False
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                # ==========================================
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            # ==========================================
            # 2. EKLENEN: ROI HESAPLAMASI (POTENTIAL ANNUAL SAVINGS)
            # ==========================================
            if lang_count > 0:
                avg_reduction = total_reduction / lang_count
                
                st.markdown("<hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
                st.subheader("💰 Potential Annual Savings (ROI Calculator)")
                st.markdown("See how much you can save annually on your LLM compute costs by switching to IQ.ai tokenization.")
                
                col_roi1, col_roi2 = st.columns([1, 2])
                
                with col_roi1:
                    monthly_spend = st.number_input("Enter your estimated monthly LLM API spend ($):", min_value=100, value=5000, step=500)
                
                with col_roi2:
                    annual_spend_current = monthly_spend * 12
                    annual_spend_iqai = annual_spend_current * (1 - (avg_reduction / 100))
                    annual_savings = annual_spend_current - annual_spend_iqai
                    
                    st.metric("🔵 Current Annual LLM Cost", f"${annual_spend_current:,.2f}")
                    st.metric("🧠 Estimated IQ.ai Annual Cost", f"${annual_spend_iqai:,.2f}")
                    st.metric("🚀 Potential Annual Savings", f"${annual_savings:,.2f}", delta=f"{avg_reduction:.1f}% Reduction", delta_color="normal")
            # ==========================================

    # ==========================================
    # TAB 3: SURGERY SCRIPTS
    # ==========================================
    with tab3:
        st.markdown("<h1 style='text-align: center;'>🛠️ Surgery Scripts</h1><p class='subtext' style='text-align: center;'>For Enterprise Integration.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        with st.expander("🩺 Step 1: Mean-Composition Surgery"):
            st.code("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nmodel = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')", language='python')
        with st.expander("📦 Step 2: 4-Bit Quantization"):
            st.code("from awq import AutoAWQForCausalLM\nmodel = AutoAWQForCausalLM.from_pretrained('model_path')", language='python')

    # ==========================================
    # TAB 4: BENCHMARK CENTER
    # ==========================================
    with tab4:
        st.markdown("<h1 style='text-align: center;'>🎯 Benchmark Center</h1><p class='subtext' style='text-align: center;'>Enterprise-grade logic and accuracy tests.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        if not api_key_input: st.warning("⚠️ Please enter your Groq API Key in the sidebar to run benchmarks.")
        else:
            st.info(f"Currently loaded with **{len(BENCHMARK_SET)}** categorized questions.")
            if st.button("🚀 Run Full Benchmark Test", type="primary"):
                client = Groq(api_key=api_key_input)
                with st.spinner("Running benchmark tests..."):
                    accuracy, categories = run_benchmark(client, BENCHMARK_SET)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.metric("🎯 Overall Benchmark Accuracy", f"{accuracy}%")
                st.markdown("</div>", unsafe_allow_html=True)
                st.subheader("📊 Category Breakdown")
                cols = st.columns(len(categories))
                for i, (cat, data) in enumerate(categories.items()):
                    with cols[i]:
                        cat_acc = round((data['correct']/data['total'])*100, 1) if data['total'] > 0 else 0
                        st.metric(f"{cat}", f"{cat_acc}%", delta=f"{data['correct']}/{data['total']} Correct")
