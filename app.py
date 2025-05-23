import streamlit as st
import numpy as np
from dabloat.db_utils import DaBloatCNNPipeline

@st.cache_resource
def load_pipeline():
 return DaBloatCNNPipeline()

classify = load_pipeline()
st.title("Audio Sentiment Analysis using Custom CNN (EMTECH)")
st.write("Upload an audio file or enter text manually.")
audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav", "mp3"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
        
    st.audio("temp_audio.wav")

    if st.button("Analyze Sentiment"):
        predicted_sentiment = classify.predict_sentiment("temp_audio.wav")
        st.markdown(f"**Predicted Voice Sentiment:** {predicted_sentiment['sentiment']}")
        st.markdown(f"**Voice Sentiment Confidence:** {predicted_sentiment['confidence'][int(np.argmax(predicted_sentiment['prediction']))] * 100:.2f}%")
        st.markdown(f"**Voice Raw Sentiment Prediction:** {predicted_sentiment['prediction']}")