import streamlit as st
import numpy as np
import tempfile, os
import nnaud as n
import librosa
import scipy.io.wavfile as wav
import noisereduce as nr

st.set_page_config(page_title="Neural Network - Audio Digit Classifier", page_icon="🎙️")
st.title("Neural Network - Audio Digit Classifier")

LABELS = {0:"zero",1:"one",2:"two",3:"three",4:"four",
          5:"five",6:"six",7:"seven",8:"eight",9:"nine"}

@st.cache_resource
def load_net():
    net = n.Neural_Net()
    net.app_loadWeights()
    return net

net = load_net()

def find_speech_window(y, sr=16000, window_size=0.1):
    window_samples = int(window_size * sr)
    energies = [
        np.sum(y[i:i+window_samples]**2)
        for i in range(0, len(y)-window_samples, window_samples//2)
    ]
    peak = np.argmax(energies) * (window_samples//2)
    start = max(0, peak - sr//2)
    return y[start:start+sr]

def predict_file(path: str) -> str:
    y, sr = librosa.load(path, sr=16000)
    y_clean = nr.reduce_noise(y=y, sr=sr)
    y_speech = find_speech_window(y_clean, sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
        wav.write(tmp2.name, 16000, (y_speech * 32767).astype(np.int16))
        feats = n.app_extract_normalized_features(file_path=tmp2.name)
        digit = net.app_inference(feats)[1]
        os.unlink(tmp2.name)
    return LABELS[digit]

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None
if "mic_result" not in st.session_state:
    st.session_state.mic_result = None

tab_mic, tab_upload = st.tabs(["🎙️ Microphone", "📁 Upload File"])

# ── Mic tab ───────────────────────────────────────────────────────────────────
with tab_mic:
    st.markdown("""
Record directly from your browser mic — no setup required.

Press the mic button, say a digit (0–9), then press stop to classify.

> Speak clearly and close to your mic.  
> The app may take a moment to load on first visit — subsequent predictions will be faster.  
""")
    audio = st.audio_input("Record a digit")

    if audio:
        audio_bytes = audio.getvalue()
        if st.session_state.get("last_audio") != audio_bytes:
            st.session_state.last_audio = audio_bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            with st.spinner("Running inference…"):
                st.session_state.mic_result = predict_file(tmp_path)
            os.unlink(tmp_path)

    if st.session_state.mic_result:
        st.success(f"*Predicted: {st.session_state.mic_result.upper()}*")

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    f = st.file_uploader("Upload a WAV file of a spoken digit (0–9)", type=["wav"])
    if f and st.button("Classify", key="cls_file"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        with st.spinner("Running inference…"):
            st.session_state.upload_result = predict_file(tmp_path)
        os.unlink(tmp_path)

    if st.session_state.upload_result:
        st.success(f"*Predicted: {st.session_state.upload_result.upper()}*")