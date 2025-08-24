import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

MODEL_NAME = "facebook/blenderbot-400M-distill"

# Load model + tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

st.title("🤖 AI Chatbot (Blenderbot)")
st.write("Chat with me, powered by Hugging Face & Streamlit!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    inputs = tokenizer([user_input], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

for speaker, text in st.session_state.history:
    st.write(f"**{speaker}:** {text}")
