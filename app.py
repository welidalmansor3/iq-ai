import streamlit as st
import requests
import os
import sentencepiece as spm
import tiktoken
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
    .big-number { font-size: 48px; font-weight: 900; text-align: center; }
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg?auth_key=1880000279-e3e53963895d4cb2b17766ad29dd2480-0-3f2ced5648a41f4923250c661dc275fd"

RENDER_URL = "https://iqai-api-iuvm.onrender.com"
MASTER_KEY = "gjai_master_key"

enc = tiktoken.encoding_for_model("gpt-4")
sp = None
DEFAULT_MODEL = "iq_ai_tokenizer.model"
if os.path.exists(DEFAULT_MODEL):
    sp = spm.SentencePieceProcessor()
    sp.load(DEFAULT_MODEL)

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


def get_api_token():
    """Render API'den token al"""
    try:
        headers = {"x-api-key": MASTER_KEY}
        payload = {
            "company_name": "DEMO_USER",
            "expiry_hours": 1,
            "allowed_sets": ["tr_ar_hi", "el_uk_fa", "ro_cs_id", "sk_hu_sl"]
        }
        res = requests.post(f"{RENDER_URL}/api/v1/auth/token", json=payload, headers=headers, timeout=30)
        if res.status_code == 200:
            return res.json().get("token")
        return None
    except:
        return None


def api_optimize(text, language, token):
    """Render API ile token optimizasyonu"""
    try:
        headers = {"x-api-key": token}
        payload = {"text": text, "language": language}
        res = requests.post(f"{RENDER_URL}/api/v1/optimize", json=payload, headers=headers, timeout=30)
        if res.status_code == 200:
            return res.json()
        return None
    except:
        return None


def local_optimize(text):
    """Yerel tokenizer ile (API olmadan)"""
    if not sp:
        return None
    gpt_tokens = enc.encode(text)
    gpt_count = len(gpt_tokens)
    iq_tokens = sp.encode(text, out_type=str)
    iq_count = len(iq_tokens)
    reduction = 100 - ((iq_count / gpt_count) * 100) if gpt_count > 0 else 0
    return {
        "gpt4_token_count": gpt_count,
        "iqai_token_count": iq_count,
        "token_reduction_percent": round(reduction, 2),
        "gpt4_tokens": [enc.decode([t]) for t in gpt_tokens],
        "iqai_tokens": iq_tokens
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
    with st.sidebar:
        st.image(LOGO_URL, use_container_width=True)
        st.markdown("---")
        st.markdown("### 📊 IQ.ai Stats")
        st.markdown("**12 Languages** | **4 SETs**")
        st.markdown("**EUR 7,000** / SET / month")
        st.markdown("---")
        st.markdown("### 📞 Contact")
        st.markdown("**Welid Almansor**")
        st.markdown("Founder & CEO, GJ.AI")
        st.markdown("📧 velitgone31@gmail.com")
        st.markdown("📱 +964 775 021 7227")
        st.markdown("🌐 [Live API](https://iqai-api-iuvm.onrender.com)")
        st.markdown("---")
        st.markdown("### 💡 No API Key Needed!")
        st.markdown("This demo uses IQ.ai's own production API directly. No Groq, no external keys required.")

    tab1, tab2, tab3 = st.tabs(["📊 Token Benchmark", "✏️ Custom Test", "🛠️ Integration"])

    # ==================== TAB 1: BENCHMARK ====================
    with tab1:
        st.markdown("<h1 style='text-align: center;'>📊 Token Benchmark</h1><p class='subtext' style='text-align: center;'>GPT-4 vs IQ.ai — Pure Language Test (No English terms)</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)

        use_api = st.checkbox("Use IQ.ai Production API (recommended)", value=True)

        if use_api:
            with st.spinner("Connecting to IQ.ai API..."):
                api_token = get_api_token()
            if api_token:
                st.success("✅ Connected to IQ.ai Production API!")
            else:
                st.warning("⚠️ API connection failed. Using local tokenizer.")
                use_api = False

        if sp or use_api:
            all_gpt = 0
            all_iq = 0

            for lang_name, lang_data in PURE_TEST_TEXTS.items():
                lang_code = lang_data["lang_code"]
                st.markdown(f"### {lang_name}")

                for size_name in ["Kısa", "Orta", "Uzun"]:
                    text = lang_data[size_name]

                    if use_api and api_token:
                        result = api_optimize(text, lang_code, api_token)
                        if result:
                            gpt_count = result.get("gpt4_token_count", 0)
                            iq_count = result.get("iqai_token_count", 0)
                            reduction = result.get("token_reduction_percent", 0)
                        else:
                            local_result = local_optimize(text)
                            if local_result:
                                gpt_count = local_result["gpt4_token_count"]
                                iq_count = local_result["iqai_token_count"]
                                reduction = local_result["token_reduction_percent"]
                            else:
                                continue
                    else:
                        local_result = local_optimize(text)
                        if local_result:
                            gpt_count = local_result["gpt4_token_count"]
                            iq_count = local_result["iqai_token_count"]
                            reduction = local_result["token_reduction_percent"]
                        else:
                            continue

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
                    st.markdown("</div>", unsafe_allow_html=True)

            # GENEL ORTALAMA
            overall = 100 - ((all_iq / all_gpt) * 100) if all_gpt > 0 else 0
            st.markdown("---")
            st.markdown(f"<div class='card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown(f"<h2>📊 GENEL ORTALAMA</h2>")
            st.markdown(f"<h1 style='color: #2575fc; font-size: 48px; font-weight: 900;'>%{overall:.1f}</h1>")
            st.markdown(f"<p class='subtext'>GPT-4: {all_gpt} tokens → IQ.ai: {all_iq} tokens</p>")
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("⚠️ IQ.ai model dosyası bulunamadı ve API bağlantısı yok.")

    # ==================== TAB 2: CUSTOM TEST ====================
    with tab2:
        st.markdown("<h1 style='text-align: center;'>✏️ Custom Text Test</h1><p class='subtext' style='text-align: center;'>Kendi metninizi test edin</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)

        custom_text = st.text_area("Metni yapıştır (Türkçe, Arapça, Hintçe):", height=150, placeholder="Buraya metninizi yapıştırın...")
        custom_lang = st.selectbox("Dil seç:", ["tr", "ar", "hi", "el", "uk", "fa", "ro", "cs", "id", "sk", "hu", "sl"], index=0)

        if st.button("🔧 Token Optimizasyon Testi", type="primary") and custom_text:
            if use_api and api_token:
                result = api_optimize(custom_text, custom_lang, api_token)
                if result:
                    gpt_count = result.get("gpt4_token_count", 0)
                    iq_count = result.get("iqai_token_count", 0)
                    reduction = result.get("token_reduction_percent", 0)
                else:
                    local_result = local_optimize(custom_text)
                    if local_result:
                        gpt_count = local_result["gpt4_token_count"]
                        iq_count = local_result["iqai_token_count"]
                        reduction = local_result["token_reduction_percent"]
                    else:
                        st.error("Test başarısız.")
                        st.stop()
            else:
                local_result = local_optimize(custom_text)
                if local_result:
                    gpt_count = local_result["gpt4_token_count"]
                    iq_count = local_result["iqai_token_count"]
                    reduction = local_result["token_reduction_percent"]
                else:
                    st.error("Local tokenizer bulunamadı.")
                    st.stop()

            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #2575fc;'>%{reduction:.1f} Token Savings</h2>")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GPT-4 Tokens", f"{gpt_count}")
            with col2:
                st.metric("IQ.ai Tokens", f"{iq_count}")
            with col3:
                st.metric("Savings", f"%{reduction:.1f}", delta="Saved", delta_color="normal")
            st.markdown("</div>", unsafe_allow_html=True)

            if sp:
                with st.expander("🔍 Token Detayları"):
                    gpt_tokens = enc.encode(custom_text)
                    iq_tokens_list = sp.encode(custom_text, out_type=str)
                    st.write(f"**GPT-4 ({gpt_count} tokens):**")
                    st.write([enc.decode([t]) for t in gpt_tokens[:30]])
                    if len(gpt_tokens) > 30:
                        st.write(f"... ({len(gpt_tokens) - 30} more)")
                    st.write(f"**IQ.ai ({iq_count} tokens):**")
                    st.write(iq_tokens_list[:30])
                    if len(iq_tokens_list) > 30:
                        st.write(f"... ({len(iq_tokens_list) - 30} more)")

    # ==================== TAB 3: INTEGRATION ====================
    with tab3:
        st.markdown("<h1 style='text-align: center;'>🛠️ Enterprise Integration</h1><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)

        st.subheader("📦 Production API")
        st.code("""
import requests

# Step 1: Generate token
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

# Step 2: Optimize text
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

Payment: Payeer
""")

        st.subheader("🌐 API Endpoints")
        st.markdown("""
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/optimize | Compare GPT-4 vs IQ.ai tokens |
| POST | /api/v1/batch/optimize | Batch optimization (1000 texts) |
| POST | /api/v1/tokenize | Direct tokenization |
| GET | /api/v1/metrics | Usage statistics |
| POST | /api/v1/auth/token | Generate JWT token |
""")

        st.subheader("📞 Contact")
        st.markdown("""
**Welid Almansor** — Founder & CEO, GJ.AI
- 📧 velitgone31@gmail.com
- 📱 +964 775 021 7227
- 🌐 https://iqai-api-iuvm.onrender.com
""")
