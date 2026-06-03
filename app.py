import streamlit as st
from groq import Groq
import os
import io
import sentencepiece as spm
import PyPDF2
from docx import Document

# LOGO URL
LOGO_URL = "https://z-cdn-media.chatglm.cn/files/97efb701-480f-41e8-a54d-d828ce634224.jpeg?auth_key=1880000279-e3e53963895d4cb2b17766ad29dd2480-0-3f2ced5648a41f4923250c661dc275fd"

# KALICI DOSYA YOLLARI (Colab'da /content/iq_ai_assets/ içine kaydeder)
ASSETS_DIR = "/content/iq_ai_assets"
CORPUS_FILE = os.path.join(ASSETS_DIR, "corpus.txt")
MODEL_PREFIX = os.path.join(ASSETS_DIR, "iq_ai_tokenizer")

# Klasörü oluştur
os.makedirs(ASSETS_DIR, exist_ok=True)

# ==========================================
# DOSYA OKUMA FONKSİYONU (PDF & DOCX)
# ==========================================
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(io.BytesIO(uploaded_file.read()))
            for para in doc.paragraphs: text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# ==========================================
# POLICY & ACCESS GATE
# ==========================================
POLICY_TEXT = """
**Terms of Service & Privacy Policy**

**1. Acceptance of Terms**
By accessing IQ.ai ("the Platform"), you agree to be bound by these Terms. If you do not agree, do not use the platform.

**2. Intellectual Property & Copyrights**
All rights, intellectual property (IP) rights, algorithms, and underlying code belong exclusively to **Welid Almansor**.
This application is developed by **GJ.AI (Great Job AI) Company**, and all copyrights are strictly reserved by GJ.AI. Unauthorized reproduction is prohibited.

**3. Logo & Trademark Protection**
The GJ.AI logo is 100% owned by GJ.AI (Great Job AI Company). Any unauthorized use, reproduction, or distribution of the logo in any context is strictly prohibited. GJ.AI reserves the full legal right to open a lawsuit and take legal action against any individual or entity that uses the logo without explicit written permission.

**4. Data & Privacy**
Chat inputs and corpus uploads are processed for generation and training. Files are saved persistently on the server runtime.

By checking the box below, you confirm your agreement.
"""

if "policy_accepted" not in st.session_state:
    st.session_state.policy_accepted = False

if not st.session_state.policy_accepted:
    st.set_page_config(page_title="IQ.ai", page_icon="🧠", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🧠 IQ.ai</h1><p style='text-align: center; color: #AAAAAA;'>Multilingual Token Optimization Engine</p><hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)
    st.warning("⚠️ **Authorization Required:** You must accept the Terms of Service.")
    with st.expander("📜 READ: Terms of Service & Privacy Policy", expanded=True):
        st.markdown(POLICY_TEXT)
    agreed = st.checkbox("I have read and I agree to the Terms of Service and Privacy Policy.")
    if st.button("🔓 Access Platform", disabled=not agreed, type="primary", use_container_width=True):
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
        api_key_input = st.text_input("Groq API Key (Free)", type="password", help="console.groq.com")
        st.markdown("---")
        st.header("📖 How to Get Free API Key")
        with st.expander("👀 Step-by-Step Guide", expanded=True):
            st.markdown("""
            👀 **Step 1:** Go to [Groq Console](https://console.groq.com)
            👀 **Step 2:** Click "Sign Up" or "Log In" (100% Free).
            👀 **Step 3:** Click **"API Keys"** on the left.
            👀 **Step 4:** Click **"Create API Key"**.
            👀 **Step 5:** Name it, click "Submit", copy the `gsk_...` code.
            👀 **Step 6:** Paste it in the box above!
            """)
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_turns = 0
            st.rerun()

    # ==========================================
    # MAIN TABS
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["🤖 Chat Engine", "🧪 Tokenizer Lab", "🛠️ Surgery Scripts"])

    # ==========================================
    # TAB 1: CHAT ENGINE
    # ==========================================
    with tab1:
        st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🧠 IQ.ai Chat Engine</h1><p style='text-align: center; color: #AAAAAA;'>Short, precise answers. Token-optimized.</p><hr style='border: 1px solid #333333;'>", unsafe_allow_html=True)

        if st.session_state.total_turns > 0:
            avg_tokens = st.session_state.total_tokens / st.session_state.total_turns
            st.caption(f"📊 Total Tokens Used: {st.session_state.total_tokens} | Avg/Messages: {avg_tokens:.1f}")

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

                        stream = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=api_messages, temperature=0.2, max_tokens=4000, stream=True)

                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")

                        message_placeholder.markdown(full_response)

                        tokens_used = int((len(prompt.split()) * 2.5) + (len(full_response.split()) * 2.0))
                        st.session_state.total_tokens += tokens_used
                        st.session_state.total_turns += 1
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.info(f"⚡ **Estimated Token Consumption:** ~{tokens_used} tokens")

                    except Exception as e:
                        st.error(f"🚫 Error: {str(e)}")

    # ==========================================
    # TAB 2: TOKENIZER LAB (PERSISTENT TRAINING)
    # ==========================================
    with tab2:
        st.subheader("🧪 Tokenizer Lab: Multilingual Unigram Training")
        st.write("Upload PDF or DOCX files. The corpus is saved, the model is really trained and saved to disk. It won't be lost when you leave the site!")

        # Check existing status
        model_exists = os.path.exists(f"{MODEL_PREFIX}.model")
        corpus_exists = os.path.exists(CORPUS_FILE)

        if model_exists:
            st.success("✅ A trained model already exists! You can load it, test it, or upload more data to retrain.")
        elif corpus_exists:
            st.info("📄 Corpus data found. Ready to train!")
        else:
            st.warning("No model or corpus found. Please upload a file to start.")

        # Upload Section
        with st.expander("📂 Upload Corpus Data (PDF/DOCX)", expanded=not model_exists):
            uploaded_file = st.file_uploader("Select file:", type=['pdf', 'docx'], key='file_uploader')

            if st.button("💾 Save to Corpus"):
                if uploaded_file:
                    with st.spinner("Extracting text..."):
                        text = extract_text_from_file(uploaded_file)
                        if text.strip():
                            # Append to the permanent corpus file
                            with open(CORPUS_FILE, "a", encoding="utf-8") as f:
                                f.write(text + "\n")
                            st.success(f"✅ Added {len(text)} chars to corpus! Total corpus size: {os.path.getsize(CORPUS_FILE)/(1024*1024):.2f} MB")
                        else:
                            st.error("Could not extract text.")

        # Train Section
        st.markdown("---")
        vocab_size = st.slider("Vocabulary Size:", min_value=1000, max_value=131072, value=32000, step=1000)

        col1, col2 = st.columns(2)
        with col1:
            train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)
        with col2:
            delete_btn = st.button("🗑️ Delete All Data", use_container_width=True)

        if delete_btn:
            import shutil
            if os.path.exists(ASSETS_DIR):
                shutil.rmtree(ASSETS_DIR)
                os.makedirs(ASSETS_DIR, exist_ok=True)
                st.success("Deleted all data. Start fresh!")
                st.rerun()

        if train_btn:
            if not corpus_exists:
                st.error("⚠️ No corpus data found. Please upload files first.")
            else:
                with st.spinner(f"Training Unigram Tokenizer with vocab size {vocab_size}. This takes time based on corpus size..."):
                    try:
                        spm.SentencePieceTrainer.train(
                            input=CORPUS_FILE,
                            model_prefix=MODEL_PREFIX,
                            vocab_size=vocab_size,
                            model_type='unigram',
                            character_coverage=1.0,
                            input_sentence_size=100000,
                            shuffle_input_sentence=True
                        )
                        st.success("🎉 Training Complete! Model saved permanently.")
                        st.rerun() # Refresh UI
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")

        # Test & Download Section
        st.markdown("---")
        if model_exists:
            sp = spm.SentencePieceProcessor()
            sp.load(f"{MODEL_PREFIX}.model")
            st.subheader("📊 Multilingual Comparison Test")

            test_sentences = {
                "🇹🇷 Turkish": "İş arayanları sahte ve hayalet ilanlardan koruyoruz.",
                "🇸🇦 Arabic": "نحن نحمي الباحثين عن عمل من الإعلانات المزيفة.",
                "🇮🇳 Hindi": "हम नौकरी तलाशने वालों को नकली विज्ञापनों से बचाते हैं।"
            }

            for lang, sentence in test_sentences.items():
                st.markdown(f"**{lang}**")
                whitespace_tokenized = sentence.split()
                sp_tokenized = sp.encode(sentence, out_type=str)
                reduction = 100 - ((len(sp_tokenized) / len(whitespace_tokenized)) * 100) if len(whitespace_tokenized) > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("*Standard Whitespace:*")
                    st.code(f"Tokens: {len(whitespace_tokenized)}")
                with col2:
                    st.markdown("*IQ.ai Unigram:*")
                    st.code(f"Tokens: {len(sp_tokenized)}")

                if reduction > 0:
                    st.info(f"💡 **Token Reduction: ~{reduction:.1f}%**")
                else:
                    st.warning("Needs more specific corpus data.")
                st.markdown("---")

            # DOWNLOAD BUTTONS
            st.subheader("⬇️ Download Trained Model")
            with open(f"{MODEL_PREFIX}.model", "rb") as f: model_data = f.read()
            with open(f"{MODEL_PREFIX}.vocab", "rb") as f: vocab_data = f.read()

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download .model", data=model_data, file_name="iq_ai_tokenizer.model", mime="application/octet-stream")
            with c2:
                st.download_button("Download .vocab", data=vocab_data, file_name="iq_ai_tokenizer.vocab", mime="application/octet-stream")

    # ==========================================
    # TAB 3: SURGERY & QUANTIZATION SCRIPTS
    # ==========================================
    with tab3:
        st.subheader("🛠️ MVP Pipeline: Surgery, Mapping & Quantization Scripts")
        st.write("These scripts are designed to run on a GPU instance. Copy and run them to apply the trained tokenizer to a real LLM.")

        with st.expander("🩺 Step 1: Mean-Composition Tokenizer Surgery"):
            st.code("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"
old_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the model you trained in the Tokenizer Lab!
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("iq_ai_tokenizer.model") # Use your downloaded file

new_tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size()) if not old_tokenizer.convert_tokens_to_ids(sp.id_to_piece(i)) == old_tokenizer.unk_token_id]

num_added = old_tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(old_tokenizer))

input_embeddings = model.get_input_embeddings().weight.data
output_embeddings = model.get_output_embeddings().weight.data

for token in new_tokens:
    token_id = old_tokenizer.convert_tokens_to_ids(token)
    subword_ids = old_tokenizer(old_tokenizer.decode(old_tokenizer.encode(token)), add_special_tokens=False)['input_ids']
    if len(subword_ids) > 0:
        input_embeddings[token_id] = input_embeddings[subword_ids].mean(dim=0)
        output_embeddings[token_id] = output_embeddings[subword_ids].mean(dim=0)

print(f"Added {num_added} tokens.")
model.save_pretrained("iq_ai_surgered_model")
old_tokenizer.save_pretrained("iq_ai_surgered_model")
            """, language='python')

        with st.expander("🧠 Step 2: Fine-Tuning"):
            st.code("""
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "corpus.txt"})

training_args = TrainingArguments(output_dir="./iq_ai_finetuned", per_device_train_batch_size=4, learning_rate=2e-4, max_steps=1000, bf16=True)
trainer = SFTTrainer(model="iq_ai_surgered_model", tokenizer=old_tokenizer, args=training_args, dataset_text_field="text", train_dataset=dataset["train"], max_seq_length=8192)
trainer.train()
trainer.save_model("iq_ai_optimized_model")
            """, language='python')

        with st.expander("📦 Step 3: 4-Bit Quantization"):
            st.code("""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "iq_ai_optimized_model"
quant_path = "iq_ai_4bit"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config, calib_data=["Türkçe verisi.", "Arabic data.", "Hindi data."])
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
            """, language='python')
