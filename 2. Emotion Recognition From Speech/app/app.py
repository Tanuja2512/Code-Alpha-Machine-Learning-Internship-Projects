import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')

# Define the emotion labels
emotion_labels = ['😊 Happy', '😢 Sad', '😡 Angry', '😨 Fear', '😐 Neutral', '🤢 Disgust', '😲 Surprise']

# Set input shape and number of classes
NUM_MFCC = 1
NUM_TIMESTEPS = 162

def preprocess_audio(audio_file):
    signal, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
    if mfccs.shape[1] < NUM_TIMESTEPS:
        mfccs = np.pad(mfccs, ((0, 0), (0, NUM_TIMESTEPS - mfccs.shape[1])), mode='constant')[:, :NUM_TIMESTEPS]
    else:
        mfccs = mfccs[:, :NUM_TIMESTEPS]
    mfccs = mfccs.reshape(1, NUM_TIMESTEPS, 1)
    return mfccs

def classify_emotion(audio_file):
    features = preprocess_audio(audio_file)
    predictions = model.predict(features)
    predicted_label = emotion_labels[np.argmax(predictions)]
    return predicted_label

# Streamlit UI
st.markdown(
    """
    <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 7vh;
            flex-direction: column;
            text-align: center;
        }
        .heading {
            background-color: #f76c6c;
            color: white;
            padding: 20px;
            border-radius: 8px;
            font-size: 40px;
            font-weight: bold;
        }
        .prediction {
            font-size: 30px;
            color: #ff6347;
            text-align: center;
            justify-content: center;
            align-items: center;
            border: 3px solid #ff6347;
            padding: 10px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Center all content on the page
st.markdown('<div class="centered">', unsafe_allow_html=True)

st.markdown('<h1 class="heading">🎵 Audio Emotion Classifier 🎤</h1>', unsafe_allow_html=True)
st.write("Upload an audio file to classify its emotion. Supported formats: WAV, MP3.")

# File uploader
audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(audio_file)
    emotion = classify_emotion("temp_audio.wav")
    st.markdown(f'<div class="prediction">The Predicted Emotion is: <strong>{emotion}</strong></div>', unsafe_allow_html=True)

# Close centered div
st.markdown('</div>', unsafe_allow_html=True)
