import streamlit as st
import requests
import os
import sentencepiece as spm
import tiktoken
import time

st.set_page_config(page_title="IQ.ai", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: #0a0a0f !important;
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #111118 !important;
        border-right: 1px solid #252530;
    }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #4a90d9 0%, #357abd 100%);
        color: white; border: none; padding: 10px 20px;
        font-weight: bold; border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(53, 122, 189, 0.4);
    }
    .stChatInput textarea {
        background-color: #1a1a24 !important;
        color: #e0e0e0 !important;
        border-color: #252530 !important;
    }
    .metric-card {
        background: #14141c;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #252530;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext { color: #808090; font-size: 13px; }
    a { color: #4a90d9 !important; }
</style>
""", unsafe_allow_html=True)

LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg?auth_key=1880000279-e3e53963895d4cb2b17766ad29dd2480-0-3f2ced5648a41f4923250c661dc275fd"

enc = tiktoken.encoding_for_model("gpt-4")
sp = None
if os.path.exists("iq_ai_tokenizer.model"):
    sp = spm.SentencePieceProcessor()
    sp.load("iq_ai_tokenizer.model")

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

POLICY_TEXT = """
**Terms of Service & Privacy Policy**

**1. Acceptance of Terms** — By accessing IQ.ai, you agree to these Terms.

**2. Intellectual Property** — All rights belong to Welid Almansor / GJ.AI Company.

**3. Zero Data Retention** — Data processed in memory only, not stored.

**4. Contact** — Welid Almansor, velitgone31@gmail.com, +964 775 021 7227
"""

if "policy_accepted" not in st.session_state:
    st.session_state.policy_accepted = False

if not st.session_state.policy_accepted:
    st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai</h1><p style='text-align: center; color: #808090;'>Multilingual Token Optimization Engine</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)
    st.warning("⚠️ Authorization Required")
    with st.expander("📜 Terms of Service", expanded=True):
        st.markdown(POLICY_TEXT)
    agreed = st.checkbox("I agree to the Terms of Service.")
    if st.button("🔓 Enter", disabled=not agreed, type="primary", use_container_width=True):
        st.session_state.policy_accepted = True
        st.rerun()

else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_gpt" not in st.session_state:
        st.session_state.total_gpt = 0
    if "total_iq" not in st.session_state:
        st.session_state.total_iq = 0
    if "total_turns" not in st.session_state:
        st.session_state.total_turns = 0

    with st.sidebar:
        st.image(LOGO_URL, use_container_width=True)
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        api_key = st.text_input("Groq API Key", type="password", help="Free key from console.groq.com")
        with st.expander("📖 How to get free API key"):
            st.markdown("1. Go to [console.groq.com](https://console.groq.com)\n2. Sign up (free)\n3. API Keys → Create\n4. Copy `gsk_...`\n5. Paste above")
        st.markdown(f"🤖 Model: `{GROQ_MODEL}`")
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_gpt = 0
            st.session_state.total_iq = 0
            st.session_state.total_turns = 0
            st.rerun()
        st.markdown("---")
        st.markdown("### 📞 Contact")
        st.markdown("Welid Almansor")
        st.markdown("GJ.AI")
        st.markdown("📧 velitgone31@gmail.com")

    # ===== MAIN AREA =====
    st.markdown("<h1 style='text-align: center;'>🧠 IQ.ai Chat</h1><p class='subtext' style='text-align: center;'>AI chat with real-time token optimization</p><hr style='border: 1px solid #252530;'>", unsafe_allow_html=True)

    # Stats bar
    if st.session_state.total_turns > 0:
        avg_savings = 100 - ((st.session_state.total_iq / st.session_state.total_gpt) * 100) if st.session_state.total_gpt > 0 else 0
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Messages", st.session_state.total_turns)
        with col2:
            st.metric("GPT-4 Tokens", st.session_state.total_gpt)
        with col3:
            st.metric("IQ.ai Tokens", st.session_state.total_iq)
        with col4:
            st.metric("Avg Savings", f"%{avg_savings:.1f}")
        st.markdown("---")

    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "token_info" in message:
                info = message["token_info"]
                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("GPT-4", f"{info['gpt_total']} tokens")
                with c2:
                    st.metric("IQ.ai", f"{info['iq_total']} tokens")
                with c3:
                    st.metric("Saved", f"%{info['reduction']:.1f}")
                st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        if not api_key:
            st.error("🚫 Please enter your Groq API key in the sidebar.")
        else:
            # User message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Call Groq API directly with requests (no SDK needed)
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        body = {
                            "model": GROQ_MODEL,
                            "messages": [{"role": "system", "content": "You are IQ.ai. Be concise and helpful."}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                            "temperature": 0.2,
                            "max_tokens": 2000
                        }

                        start_time = time.time()
                        response = requests.post(GROQ_URL, json=body, headers=headers, timeout=60)
                        end_time = time.time()

                        if response.status_code == 200:
                            data = response.json()
                            ai_response = data["choices"][0]["message"]["content"]
                            latency = int((end_time - start_time) * 1000)

                            st.markdown(ai_response)

                            # Token analysis
                            gpt_prompt = len(enc.encode(prompt))
                            gpt_response = len(enc.encode(ai_response))
                            gpt_total = gpt_prompt + gpt_response

                            iq_total = 0
                            reduction = 0
                            if sp:
                                iq_prompt = len(sp.encode(prompt))
                                iq_response = len(sp.encode(ai_response))
                                iq_total = iq_prompt + iq_response
                                if gpt_total > 0:
                                    reduction = round(100 - ((iq_total / gpt_total) * 100), 2)

                            st.session_state.total_gpt += gpt_total
                            st.session_state.total_iq += iq_total
                            st.session_state.total_turns += 1

                            token_info = {
                                "gpt_total": gpt_total,
                                "iq_total": iq_total,
                                "reduction": reduction,
                                "latency": latency
                            }

                            # Show token comparison
                            st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown(f"<p class='subtext' style='text-align: center;'>⚡ {latency}ms | Token Optimization</p>", unsafe_allow_html=True)
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("GPT-4", f"{gpt_total} tokens")
                            with c2:
                                st.metric("IQ.ai", f"{iq_total} tokens")
                            with c3:
                                st.metric("Saved", f"%{reduction:.1f}")
                            st.markdown("</div>", unsafe_allow_html=True)

                            with st.expander("🔍 Token Details"):
                                st.write(f"**Input:** GPT-4={gpt_prompt} → IQ.ai={iq_prompt if sp else 'N/A'}")
                                st.write(f"**Output:** GPT-4={gpt_response} → IQ.ai={iq_response if sp else 'N/A'}")
                                st.write(f"**Total:** GPT-4={gpt_total} → IQ.ai={iq_total}")
                                st.write(f"**Reduction:** %{reduction}")
                                st.write(f"**Latency:** {latency}ms")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": ai_response,
                                "token_info": token_info
                            })

                        else:
                            error_data = response.json()
                            error_msg = error_data.get("error", {}).get("message", "Unknown error")
                            st.error(f"🚫 API Error: {error_msg}")

                    except requests.exceptions.Timeout:
                        st.error("🚫 Request timed out. Try again.")
                    except Exception as e:
                        st.error(f"🚫 Error: {str(e)}")
