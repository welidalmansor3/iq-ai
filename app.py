import streamlit as st
from groq import Groq
import os
import sentencepiece as spm
import tiktoken
import re
import time
import json

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
GROQ_MODEL = "openai/gpt-oss-20b"

enc = tiktoken.encoding_for_model("gpt-4")
sp = None
if os.path.exists(DEFAULT_MODEL):
    sp = spm.SentencePieceProcessor()
    sp.load(DEFAULT_MODEL)

# --- POLICY ---
POLICY_TEXT = """
**Terms of Service & Privacy Policy**

**1. Acceptance of Terms**
By accessing IQ.ai, you agree to be bound by these Terms.

**2. Intellectual Property**
All rights, IP, algorithms, and code belong to Welid Almansor / GJ.AI Company.

**3. Zero Data Retention**
IQ.ai operates under Zero Data Retention. Data processed in memory only.

**4. Contact**
Welid Almansor, Founder, GJ.AI
Email: velitgone31@gmail.com
Phone: +964 775 021 7227
"""

# --- SAF DIL TEST METINLERI ---
PURE_TEST_TEXTS = {
    "🇹🇷 Türkçe": {
        "lang_code": "tr",
        "Kısa": "Yapay zeka modelleri metinleri işlerken kelimeleri alt parçalara böler ve bu durum maliyetleri artırır.",
        "Orta": "Yapay zeka kavramı ilk kez 1956 yılında Dartmouth Konferansında ortaya atılmıştır. O dönemde bilim insanları, bilgisayarların düşünme yeteneği kazanabileceğine inanıyordu. Yıllar geçtikçe bu alanda büyük ilerlemeler kaydedildi ve bugün yapay zeka sağlık, eğitim, tarım ve sanayi gibi birçok sektörde kullanılmaktadır.",
        "Uzun": "Yapay zeka kavramı ilk kez 1956 yılında Dartmouth Konferansında ortaya atılmıştır. O dönemde bilim insanları, bilgisayarların düşünme yeteneği kazanabileceğine inanıyordu. Yıllar geçtikçe bu alanda büyük ilerlemeler kaydedildi ve bugün yapay zeka sağlık, eğitim, tarım ve sanayi gibi birçok sektörde kullanılmaktadır. Ancak yapay zekanın hızlı gelişimi beraberinde etik soruları da getirmektedir. Yapay zekanın geleceği hakkında ne düşünüyorsunuz ve bu teknolojinin insanlığa en büyük faydası ne olabilir?"
    },
    "🇸🇦 Arapça": {
        "lang_code": "ar",
        "Kısa": "يجب على جميع الأطراف الالتزام بشروط العقد المذكورة أعلاه دون أي تأخير.",
        "Orta": "ظهر مفهوم الذكاء الاصطناعي لأول مرة في عام 1956 خلال مؤتمر دارتموث. في ذلك الوقت، اعتقد العلماء أن أجهزة الحاسوب يمكن أن تكتسب القدرة على التفكير. على مر السنين، تحقق تقدم كبير في هذا المجال واليوم يُستخدم الذكاء الاصطناعي في الصحة والتعليم والزراعة والصناعة.",
        "Uzun": "ظهر مفهوم الذكاء الاصطناعي لأول مرة في عام 1956 خلال مؤتمر دارتموث. في ذلك الوقت، اعتقد العلماء أن أجهزة الحاسوب يمكن أن تكتسب القدرة على التفكير. على مر السنين، تحقق تقدم كبير في هذا المجال واليوم يُستخدم الذكاء الاصطناعي في الصحة والتعليم والزراعة والصناعة. ومع ذلك، فإن التطور السريع للذكاء الاصطناعي يثير تساؤلات أخلاقية كثيرة. ما هو رأيك في مستقبل الذكاء الاصطناعي وماذا يمكن أن يكون أكبر فائدة لهذه التقنية للبشرية؟"
    },
    "🇮🇳 Hintçe": {
        "lang_code": "hi",
        "Kısa": "कृत्रिम बुद्धिमत्ता मॉडल पाठ को संसाधित करते समय शब्दों को छोटे टुकड़ों में तोड़ देते हैं।",
        "Orta": "कृत्रिम बुद्धिमत्ता की अवधारणा पहली बार 1956 में डार्टमाउथ सम्मेलन में प्रस्तुत की गई थी। उस समय, वैज्ञानिकों का मानना था कि संगणक सोचने की क्षमता प्राप्त कर सकते हैं। वर्षों बीतने के साथ, इस क्षेत्र में भारी प्रगति हुई और आज कृत्रिम बुद्धिमत्ता का उपयोग स्वास्थ्य, शिक्षा, कृषि और उद्योग में किया जा रहा है।",
        "Uzun": "कृत्रिम बुद्धिमत्ता की अवधारणा पहली बार 1956 में डार्टमाउथ सम्मेलन में प्रस्तुत की गई थी। उस समय, वैज्ञानिकों का मानना था कि संगणक सोचने की क्षमता प्राप्त कर सकते हैं। वर्षों बीतने के साथ, इस क्षेत्र में भारी प्रगति हुई और आज कृत्रिम बुद्धिमत्ता का उपयोग स्वास्थ्य, शिक्षा, कृषि और उद्योग में किया जा रहा है। हालांकि, कृत्रिम बुद्धिमत्ता के तेजी से विकास से कई नैतिक प्रश्न उत्पन्न होते हैं। आप कृत्रिम बुद्धिमत्ता के भविष्य के बारे में क्या सोचते हैं और इस तकनीक का मानवता के लिए सबसे बड़ा लाभ क्या हो सकता है?"
    }
}

if "policy_accepted" not in st.session_state:
    st.session_state.policy_accepted = False

if not st.session_state.policy_accepted:
    st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai</h1><p style='text-align: center; color: #a0a0a0;'>Multilingual Token Optimization Engine</p><hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)
    st.warning("⚠️ **Authorization Required:** You must accept the Terms of Service.")
    with st.expander("📜 Terms of Service & Privacy Policy", expanded=True):
        st.markdown(POLICY_TEXT)
    agreed = st.checkbox("I have read and I agree to the Terms of Service and Privacy Policy.")
    if st.button("🔓 Access Platform", disabled=not agreed, type="primary", use_container_width=True):
        st.session_state.policy_accepted = True
        st.rerun()

else:
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
        with st.expander("📖 Get Free API Key"):
            st.markdown("1. Go to [Groq Console](https://console.groq.com)\n2. Sign Up → API Keys → Create\n3. Copy `gsk_...`")
        st.markdown(f"🤖 Active Model: `{GROQ_MODEL}`")
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_turns = 0
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🤖 Chat Engine", "📊 Token Benchmark", "🛠️ Integration"])

    # ==================== TAB 1: CHAT ====================
    with tab1:
        st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai Chat Engine</h1><p class='subtext' style='text-align: center;'>Token-optimized multilingual AI</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
        if st.session_state.total_turns > 0:
            st.caption(f"📊 Total Tokens: {st.session_state.total_tokens} | Avg: {st.session_state.total_tokens/st.session_state.total_turns:.1f}")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            if not api_key_input:
                st.error("🚫 Please enter your Groq API key in the sidebar.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    try:
                        client = Groq(api_key=api_key_input)
                        sys_pr = {"role": "system", "content": "You are IQ.ai. Be concise and accurate."}

                        start_time = time.time()
                        stream = client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=[sys_pr] + st.session_state.messages,
                            temperature=0.2,
                            max_tokens=4000,
                            stream=True
                        )
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
                            if gpt_total > 0:
                                token_reduction = round(100 - ((iq_total / gpt_total) * 100), 2)

                        st.session_state.total_tokens += gpt_total
                        st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("💰 Token Saved", f"%{token_reduction}")
                        with col2:
                            st.metric("⚡ Latency", f"{latency_ms}ms")
                        with col3:
                            st.metric("📊 GPT-4 vs IQ.ai", f"{gpt_total} → {iq_total}")

                        with st.expander("🔍 Detailed Token Breakdown"):
                            st.write(f"**GPT-4 Prompt:** {gpt_prompt_tokens} tokens")
                            st.write(f"**IQ.ai Prompt:** {iq_prompt_tokens if sp else 'N/A'} tokens")
                            st.write(f"**GPT-4 Response:** {gpt_response_tokens} tokens")
                            st.write(f"**IQ.ai Response:** {iq_response_tokens if sp else 'N/A'} tokens")
                            st.write(f"**Total Reduction:** %{token_reduction}")

                    except Exception as e:
                        st.error(f"🚫 Error: {str(e)}")

    # ==================== TAB 2: TOKEN BENCHMARK ====================
    with tab2:
        st.markdown("<h1 style='text-align: center;'>📊 Token Benchmark</h1><p class='subtext' style='text-align: center;'>GPT-4 vs IQ.ai — Pure Language Test</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)

        if sp:
            st.success("✅ IQ.ai Tokenizer Active!")

            # --- KISIM 1: Sabit Test Metinleri ---
            st.subheader("📋 Sabit Benchmark (3 Dil × 3 Boyut)")

            all_gpt = 0
            all_iq = 0

            for lang_name, lang_data in PURE_TEST_TEXTS.items():
                lang_code = lang_data["lang_code"]
                st.markdown(f"### {lang_name}")

                for size_name in ["Kısa", "Orta", "Uzun"]:
                    text = lang_data[size_name]

                    gpt_tokens = enc.encode(text)
                    gpt_count = len(gpt_tokens)

                    iq_tokens = sp.encode(text, out_type=str)
                    iq_count = len(iq_tokens)

                    reduction = 100 - ((iq_count / gpt_count) * 100) if gpt_count > 0 else 0

                    all_gpt += gpt_count
                    all_iq += iq_count

                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"**{size_name}** — {lang_name}")
                    st.markdown(f"<p class='subtext'><i>\"{text[:80]}...\"</i></p>", unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("GPT-4", f"{gpt_count} tokens")
                    with col2:
                        st.metric("IQ.ai", f"{iq_count} tokens")
                    with col3:
                        st.metric("Tasarruf", f"%{reduction:.1f}", delta="Saved", delta_color="normal")

                    with st.expander("🔍 Token Detayları"):
                        st.write("**GPT-4:**", [enc.decode([t]) for t in gpt_tokens[:20]], "..." if len(gpt_tokens) > 20 else "")
                        st.write("**IQ.ai:**", iq_tokens[:20], "..." if len(iq_count) > 20 else "")
                    st.markdown("</div>", unsafe_allow_html=True)

            # --- GENEL ORTALAMA ---
            overall = 100 - ((all_iq / all_gpt) * 100) if all_gpt > 0 else 0
            st.markdown("---")
            st.markdown(f"<div class='card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown(f"<h2>📊 GENEL ORTALAMA</h2>")
            st.markdown(f"<h1 style='color: #2575fc;'>%{overall:.1f}</h1>")
            st.markdown(f"<p class='subtext'>GPT-4: {all_gpt} tokens → IQ.ai: {all_iq} tokens</p>")
            st.markdown("</div>", unsafe_allow_html=True)

            # --- KISIM 2: Kendi Metnini Test Et ---
            st.markdown("---")
            st.subheader("✏️ Kendi Metnini Test Et")
            custom_text = st.text_area("Metni yapıştır (Türkçe, Arapça, Hintçe):", height=150)
            custom_lang = st.selectbox("Dil seç:", ["tr", "ar", "hi"], index=0)

            if st.button("🔧 Test Et", type="primary") and custom_text:
                gpt_tokens = enc.encode(custom_text)
                gpt_count = len(gpt_tokens)
                iq_tokens = sp.encode(custom_text, out_type=str)
                iq_count = len(iq_tokens)
                reduction = 100 - ((iq_count / gpt_count) * 100) if gpt_count > 0 else 0

                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("GPT-4", f"{gpt_count} tokens")
                with col2:
                    st.metric("IQ.ai", f"{iq_count} tokens")
                with col3:
                    st.metric("Tasarruf", f"%{reduction:.1f}", delta="Saved", delta_color="normal")
                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("🔍 Token Detayları"):
                    st.write("**GPT-4 parçalar:**", [enc.decode([t]) for t in gpt_tokens[:30]])
                    st.write("**IQ.ai parçalar:**", iq_tokens[:30])

        else:
            st.warning("⚠️ IQ.ai model dosyası bulunamadı.")

    # ==================== TAB 3: INTEGRATION ====================
    with tab3:
        st.markdown("<h1 style='text-align: center;'>🛠️ Integration Guide</h1><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)

        st.subheader("📦 Enterprise API")
        st.code("""
import requests

# Token üret
headers = {"x-api-key": "<master_key>"}
payload = {
    "company_name": "YOUR_COMPANY",
    "expiry_hours": 720,
    "allowed_sets": ["tr_ar_hi"]
}
res = requests.post(
    "https://iqai-api-iuvm.onrender.com/api/v1/auth/token",
    json=payload, headers=headers
)
token = res.json()["token"]

# Optimize et
headers = {"x-api-key": token}
payload = {"text": "Arapça metin...", "language": "ar"}
res = requests.post(
    "https://iqai-api-iuvm.onrender.com/api/v1/optimize",
    json=payload, headers=headers
)
print(res.json())
# {"gpt4_token_count": 46, "iqai_token_count": 15, "token_reduction_percent": 67.4}
""", language="python")

        st.subheader("💰 Pricing")
        st.markdown("""
| Edition | Scope | Monthly Fee |
|---------|-------|-------------|
| Single SET | 3 languages | EUR 7,000 |
| Complete Engine | All 12 languages | EUR 30,000 |
""")

        st.subheader("📞 Contact")
        st.markdown("""
**Welid Almansor** — Founder & CEO, GJ.AI
- Email: velitgone31@gmail.com
- Phone: +964 775 021 7227
- API: https://iqai-api-iuvm.onrender.com
""")
