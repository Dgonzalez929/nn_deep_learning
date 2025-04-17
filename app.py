import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
import tempfile
import traceback

# Streamlit page config
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")

# Model paths
MODEL_PATHS = {
    "CNN": "models/best_cnn_model.h5",
    "RNN": "models/best_Rnn_model.h5",
    "SVM": "models/svm_model.joblib"
}

# Validation accuracy for each model
MODEL_ACCURACY = {
    "CNN": 0.8016,
    "RNN": 0.8522,
    "SVM": 0.9145
}

# List of genres
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_audio_from_uploaded_file(uploaded_file, duration=30):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    y, sr = librosa.load(tmp_path, sr=None, duration=duration, mono=True)
    return y, sr

def extract_features_tabular(uploaded_file):
    y, sr = load_audio_from_uploaded_file(uploaded_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    stats = [
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(rolloff), np.std(rolloff),
        np.mean(zero_crossing), np.std(zero_crossing)
    ]

    features = np.concatenate([
        mfcc_mean,              # 20
        mfcc_std,               # 20
        chroma_mean[:10],       # 10 primeros
        stats                   # 8 estadísticos
    ])

    return features.reshape(1, -1)

def extract_features_cnn(uploaded_file, n_mfcc=20, max_len=130):
    y, sr = load_audio_from_uploaded_file(uploaded_file)
    y, _ = librosa.effects.trim(y)
    samples_per_segment = int(3 * sr)
    if len(y) < samples_per_segment:
        return None
    segment = y[:samples_per_segment]
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T[np.newaxis, ..., np.newaxis]

def extract_features_rnn(uploaded_file, segment_duration=3, n_mfcc=20, max_len=130):
    y, sr = load_audio_from_uploaded_file(uploaded_file)
    y, _ = librosa.effects.trim(y)
    samples_per_segment = int(segment_duration * sr)
    if len(y) < samples_per_segment:
        return None
    segment = y[:samples_per_segment]
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T[np.newaxis, ...]

def plot_waveform(y, sr):
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

def plot_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    ax.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

def plot_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    ax.set_title('Chroma Feature')
    fig.colorbar(img, ax=ax)
    return fig

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🎶 Music Genre Classifier</h1>
    <p style='text-align: center;'>Powered by CNN, RNN and SVM models</p>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    y_preview, sr_preview = load_audio_from_uploaded_file(uploaded_file)

    st.subheader("📈 Audio Visualizations")
    st.pyplot(plot_waveform(y_preview, sr_preview))
    st.pyplot(plot_mel_spectrogram(y_preview, sr_preview))
    st.pyplot(plot_chroma(y_preview, sr_preview))

    with st.spinner("🔍 Extracting features and predicting..."):
        try:
            st.subheader("📊 Predictions by Model")

            for model_name, model_path in MODEL_PATHS.items():
                ext = os.path.splitext(model_path)[1]

                if ext == ".h5":
                    if model_name == "CNN":
                        features = extract_features_cnn(uploaded_file)
                    elif model_name == "RNN":
                        features = extract_features_rnn(uploaded_file)
                        if features is None:
                            st.warning("Audio is too short for the RNN model (requires at least 3 seconds).")
                            continue
                    model = tf.keras.models.load_model(model_path)
                    probs = model.predict(features)[0]
                    pred_index = np.argmax(probs)
                    pred = GENRES[pred_index]
                    confidence = probs[pred_index]

                elif ext in [".joblib", ".pkl"]:
                    features = extract_features_tabular(uploaded_file)

                    # Load and apply scaler for SVM
                    scaler = joblib.load("models/scaler_svm.joblib")
                    features = scaler.transform(features)

                    model = joblib.load(model_path)
                    pred_index = int(model.predict(features)[0])
                    pred = GENRES[pred_index]
                    confidence = 1.0

                st.markdown(f"""
                    ### 🤖 {model_name}
                    - **Validation Accuracy:** `{MODEL_ACCURACY[model_name]*100:.2f}%`
                    - **Predicted Genre:** 🎵 **{pred}**
                    - **Confidence:** `{confidence*100:.2f}%`
                """)

        except Exception as e:
            st.error(f"🚨 Error processing the file:\n\n{e}")
            st.text("🛠️ Debug Info:")
            st.text(traceback.format_exc())