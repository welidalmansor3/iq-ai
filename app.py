import streamlit as st
from groq import Groq
import os
import sentencepiece as spm
import tiktoken
import re
import time
import json
from duckduckgo_search import DDGS

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

LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg?auth_key=1880000279-e3e53963895d4cb2b17766ad29dd2480-0-3f2ced5648a41f4923250c661dc275fd"
DEFAULT_MODEL = "iq_ai_tokenizer.model"

enc = tiktoken.encoding_for_model("gpt-4")
sp = None
if os.path.exists(DEFAULT_MODEL):
    sp = spm.SentencePieceProcessor()
    sp.load(DEFAULT_MODEL)

POLICY_TEXT = """
**Terms of Service & Privacy Policy**
**1. Acceptance of Terms** By accessing IQ.ai ("the Platform"), you agree to be bound by these Terms.
**2. Intellectual Property & Copyrights** All rights, IP, algorithms, and code belong exclusively to **Welid Almansor / GJ.AI Company**.
**3. Logo & Trademark Protection** The GJ.AI logo is 100% owned. Unauthorized use is strictly prohibited and subject to lawsuit.
**4. Data Privacy** Chat inputs are processed securely. Trained models are hosted for demonstration.
By checking the box below, you confirm your agreement.
"""

def search_web(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = [{"title": r.get("title",""), "url": r.get("href",""), "body": r.get("body","")} for r in ddgs.text(query, max_results=max_results)]
            return results if results else []
    except: return []

def run_integrity_check(client, question, answer):
    web_context = search_web(question)
    context_str = json.dumps(web_context, indent=2)
    
    prompt = f"""
    Question: {question}
    Model Answer: {answer}
    Web Evidence: {context_str}
    
    Based on the Web Evidence, calculate strictly:
    1. source_match (0-100): Does the Web Evidence explicitly support the answer's claims?
    2. consistency (0-100): Is the answer internally logical without contradictions?
    3. claim_verification (0-100): Are the specific factual claims in the answer verifiable?
    4. hallucination_score (0-100): Amount of fabricated info (0=none, 100=total fake).
    
    Return ONLY valid JSON:
    {{
        "source_match": <int>,
        "consistency": <int>,
        "claim_verification": <int>,
        "hallucination_score": <int>,
        "reason": "<short reason>"
    }}
    """
    try:
        result = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":prompt}], temperature=0)
        text = re.sub(r'```json|```', '', result.choices[0].message.content.strip())
        data = json.loads(text)
        
        source_match = data.get("source_match", 50)
        consistency = data.get("consistency", 50)
        claim_ver = data.get("claim_verification", 50)
        hall_score = data.get("hallucination_score", 50)
        
        confidence = int((source_match * 0.4) + (consistency * 0.3) + (claim_ver * 0.3))
        
        return {
            "confidence": min(100, max(0, confidence)),
            "source_match": source_match,
            "consistency": consistency,
            "claim_verification": claim_ver,
            "hallucination_score": hall_score,
            "risk": "HIGH" if hall_score > 60 else "MEDIUM" if hall_score > 30 else "LOW",
            "reason": data.get("reason", "N/A"),
            "sources": web_context
        }
    except Exception as e:
        return {"confidence": 50, "source_match": 50, "consistency": 50, "claim_verification": 50, "hallucination_score": 50, "risk": "PARSE_ERR", "reason": str(e), "sources": []}

def generate_health_report(token_reduction, confidence, hallucination_score, latency):
    token_score = min(100, token_reduction * 2) if token_reduction > 0 else 0
    cost_score = token_score
    safety_score = 100 - hallucination_score
    latency_score = max(0, 100 - (latency / 50))
    
    overall = int((confidence * 0.30) + (safety_score * 0.30) + (token_score * 0.15) + (cost_score * 0.15) + (latency_score * 0.10))
    return {"Token Efficiency": token_score, "Cost Score": cost_score, "Safety Score": safety_score, "Latency Score": latency_score, "Confidence": confidence, "Overall": overall}

RAW_BENCHMARK = """
General|Türkiye'nin başkenti nedir?|Ankara
General|Dünyanın en büyük okyanusu nedir?|Pasifik
General|Suyun kimyasal formülü nedir?|H2O
General|Who wrote Romeo and Juliet?|Shakespeare
General|Fransa'nın başkenti neresidir?|Paris
General|Dünyanın en büyük kıtası hangisidir?|Asya
General|Mars hangi gezegenin yanındadır?|Jupiter
General|Isik hizi nedir?|300000 km/s
General|Demirin sembolu nedir?|Fe
General|Einsteins famous equation?|E=mc2
Math|2+2 kaç eder?|4
Math|10 * 5 kaçtır?|50
Math|15 - 7 kaçtır?|8
Math|100 / 4 kaçtır?|25
Math|3 ussu 3 kaçtır?|27
Math|Karekök 144 kaçtır?|12
Math|What is 20% of 200?|40
Math|Prime number after 7?|11
Math|Factorial of 5?|120
Math|Area of 5x6 rectangle?|30
Logic|Tüm kediler hayvansa, hayvanlar nefes alıyorsa, kediler nefes alır mı?|Evet
Logic|If it rains, the ground gets wet. The ground is wet. Did it rain?|Not necessarily
Logic|All A are B. All B are C. Is all A C?|Yes
Logic|Syllogism: No fish can fly. Herring is a fish. Can herring fly?|No
Logic|If X>5 and X<10, is X=4 possible?|No
Logic|True or False: If it is snowing, it is cold.|True
Logic|Contrapositive: If not B then not A?|If A then B
Logic|If P implies Q, and P is false, is Q true?|Unknown
Logic|A is taller than B. B is taller than C. Is A taller than C?|Yes
Logic|If all birds fly and penguins are birds, do penguins fly?|Paradox
Multilingual|How do you say 'Hello' in Spanish?|Hola
Multilingual|What is 'Thank you' in French?|Merci
Multilingual|كيف تقول 'ماء' بالإنجليزية؟|Water
Multilingual|How to say 'Goodbye' in German?|Auf Wiedersehen
Multilingual|What is 'Book' in Italian?|Libro
Multilingual|Comment dire 'Maison' en Anglais?|House
Multilingual|Wie sagt man 'Liebe' auf Englisch?|Love
Multilingual|What is 'Sun' in Arabic?|شمس
Multilingual|What is 'Moon' in Turkish?|Ay
Multilingual|What is 'Star' in Hindi?|तारा
"""

BENCHMARK_SET = []
for line in RAW_BENCHMARK.strip().split('\n'):
    parts = line.split('|')
    if len(parts) == 3:
        BENCHMARK_SET.append({"category": parts[0].strip(), "question": parts[1].strip(), "answer": parts[2].strip()})

def run_benchmark(client, benchmark_set):
    categories = {}
    correct_total = 0
    for item in benchmark_set:
        try:
            result = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"user","content":item["question"]}], temperature=0)
            answer = result.choices[0].message.content
            cat = item["category"]
            if cat not in categories: categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if item["answer"].lower() in answer.lower():
                categories[cat]["correct"] += 1
                correct_total += 1
        except: pass
    accuracy = round(correct_total/len(benchmark_set)*100, 2) if benchmark_set else 0
    return accuracy, categories

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

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Chat Engine", "📊 Token Comparison", "🛠️ Surgery Scripts", "🎯 Benchmark Center"])

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
                        
                        gpt_prompt_tokens = len(enc.encode(prompt))
                        gpt_response_tokens = len(enc.encode(full_response))
                        gpt_total = gpt_prompt_tokens + gpt_response_tokens
                        
                        iq_total = 0
                        token_reduction = 0
                        if sp:
                            iq_prompt_tokens = len(sp.encode(prompt))
                            iq_response_tokens = len(sp.encode(full_response))
                            iq_total = iq_prompt_tokens + iq_response_tokens
                            if gpt_total > 0: token_reduction = round(100 - ((iq_total / gpt_total) * 100), 2)
                        
                        st.session_state.total_tokens += gpt_total
                        st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        with st.spinner("🔍 Running Integrity Check (Source & Claim Verification)..."):
                            integrity = run_integrity_check(client, prompt, full_response)
                            report = generate_health_report(token_reduction=token_reduction, confidence=integrity["confidence"], hallucination_score=integrity["hallucination_score"], latency=latency_ms)
                            
                            st.markdown("---")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("🛡 Confidence", f"{integrity['confidence']}%")
                            with col2: st.metric("🚨 Hall. Risk", integrity['risk'])
                            with col3: st.metric("💰 Token Saved", f"%{token_reduction}")
                            with col4: st.metric("🏆 Overall Health", f"{report['Overall']}")

                            with st.expander("🔬 Detailed AI Integrity Analysis"):
                                st.subheader("1. Core Metrics")
                                sc1, sc2, sc3 = st.columns(3)
                                with sc1: st.metric("Source Match", f"{integrity['source_match']}%")
                                with sc2: st.metric("Consistency", f"{integrity['consistency']}%")
                                with sc3: st.metric("Claim Verification", f"{integrity['claim_verification']}%")
                                
                                st.subheader("2. Hallucination Report")
                                st.json({"Risk Level": integrity['risk'], "Hallucination Score": f"{integrity['hallucination_score']}/100", "Reason": integrity['reason']})
                                
                                st.subheader("3. Web Sources & Evidence")
                                if integrity['sources']:
                                    for src in integrity['sources']:
                                        st.markdown(f"**{src.get('title', 'No Title')}**")
                                        st.markdown(f"🔗 [{src.get('url', '#')}]({src.get('url', '#')})")
                                        st.caption(src.get('body', 'No snippet'))
                                        st.markdown("---")
                                else:
                                    st.warning("No web sources found.")
                                
                                st.subheader("🩺 System Health Report (Weighted)")
                                hcol1, hcol2, hcol3 = st.columns(3)
                                with hcol1:
                                    st.metric("Token Efficiency (x0.15)", f"{report['Token Efficiency']}")
                                    st.metric("Cost Score (x0.15)", f"{report['Cost Score']}")
                                with hcol2:
                                    st.metric("Safety Score (x0.30)", f"{report['Safety Score']}")
                                    st.metric("Confidence (x0.30)", f"{report['Confidence']}")
                                with hcol3:
                                    st.metric("Latency Score (x0.10)", f"{report['Latency Score']}")
                                    st.metric("Overall System Score", f"{report['Overall']}")

                    except Exception as e: st.error(f"🚫 Error: {str(e)}")

    with tab2:
        st.markdown("<h1 style='text-align: center;'>📊 Live Token Tax Benchmark</h1><p class='subtext' style='text-align: center;'>GPT-4 Tokenizer vs IQ.ai Custom Tokenizer.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        if sp:
            st.success("✅ IQ.ai Custom Multilingual Tokenizer is Active!")
            test_sentences = {"🇹🇷 Turkish": "İş arayanları sahte ve hayalet ilanlardan koruyoruz.", "🇸🇦 Arabic": "نحن نحمي الباحثين عن عمل من الإعلانات المزيفة.", "🇮🇳 Hindi": "हम नौकरी तलाशने वालों को नकली विज्ञापनों से बचाते हैं।"}
            for lang, sentence in test_sentences.items():
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{lang}**")
                st.markdown(f"<p class='subtext'><i>\"{sentence}\"</i></p>", unsafe_allow_html=True)
                gpt_tokens = enc.encode(sentence); gpt_count = len(gpt_tokens)
                iq_tokens = sp.encode(sentence, out_type=str); iq_count = len(iq_tokens)
                reduction = 100 - ((iq_count / gpt_count) * 100) if gpt_count > 0 else 0
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("GPT-4 Tokenizer", f"{gpt_count} Tokens")
                with col2: st.metric("IQ.ai Unigram", f"{iq_count} Tokens")
                with col3: st.metric("Token Reduction", f"%{reduction:.1f}", delta="Saved", delta_color="normal")
                with st.expander("🔍 GPT-4 vs IQ.ai Token Breakdown"):
                    st.write("**GPT-4 Fragments:**", [enc.decode([t]) for t in gpt_tokens])
                    st.write("**IQ.ai Fragments:**", iq_tokens)
                st.markdown("</div>", unsafe_allow_html=True)
        else: st.warning("⚠️ Trained model (`iq_ai_tokenizer.model`) not found in repository.")

    with tab3:
        st.markdown("<h1 style='text-align: center;'>🛠️ Surgery Scripts</h1><p class='subtext' style='text-align: center;'>For Enterprise Integration.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        with st.expander("🩺 Step 1: Mean-Composition Surgery"):
            st.code("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nmodel = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')\nold_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')\n# Load your trained IQ.ai tokenizer and add tokens...\n# model.resize_token_embeddings(len(old_tokenizer))", language='python')
        with st.expander("📦 Step 2: 4-Bit Quantization"):
            st.code("from awq import AutoAWQForCausalLM\nmodel = AutoAWQForCausalLM.from_pretrained('model_path')\nmodel.quantize(tokenizer, quant_config={ 'zero_point': True, 'q_group_size': 128, 'w_bit': 4 })", language='python')

    with tab4:
        st.markdown("<h1 style='text-align: center;'>🎯 Benchmark Center</h1><p class='subtext' style='text-align: center;'>Enterprise-grade logic and accuracy tests.</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        if not api_key_input: st.warning("⚠️ Please enter your Groq API Key in the sidebar to run benchmarks.")
        else:
            st.info(f"Currently loaded with **{len(BENCHMARK_SET)}** categorized questions (General, Math, Logic, Multilingual).")
            if st.button("🚀 Run Full Benchmark Test", type="primary"):
                client = Groq(api_key=api_key_input)
                with st.spinner("Running benchmark tests... This may take a minute."):
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