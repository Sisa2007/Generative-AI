
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load model
@st.cache_resource
def load_rnn_model():
    model = load_model("simple_rnn_model (1).h5")
    return model

model = load_rnn_model()

# Dummy tokenizer (replace with actual tokenizer or logic)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(["this is a sample text", "another sample"])

st.title("🧠 Simple RNN Model Prediction")
st.write("Upload a sentence to get prediction using your `.h5` model.")

user_input = st.text_input("Enter a sentence:")

if user_input:
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_seq)
    st.subheader("🔮 Model Prediction:")
    st.write(prediction)
