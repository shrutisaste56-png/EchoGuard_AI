
import gradio as gr
import numpy as np
import pandas as pd
import joblib
import librosa
import os
import html
from spafe.features.lfcc import lfcc


# ==========================================
# 1. LOAD TRAINED AI MODEL
# ==========================================

MODEL_PATH = "voice_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "voice_model.pkl not found. Run model_trainer.py first."
    )

model = joblib.load(MODEL_PATH)

print("EchoGuard AI Model Loaded Successfully!")


# ==========================================
# 2. EXTRACT LFCC FEATURES
# ==========================================

def extract_audio_features(audio_file_path):

    y, sr = librosa.load(
        audio_file_path,
        sr=16000,
        mono=True
    )

    y_trimmed, _ = librosa.effects.trim(
        y,
        top_db=20
    )

    if len(y_trimmed) < 16000:
        y_trimmed = np.pad(
            y_trimmed,
            (0, 16000 - len(y_trimmed)),
            mode="constant"
        )

    lfccs = lfcc(
        sig=y_trimmed,
        fs=sr,
        num_ceps=20,
        pre_emph=0.97
    )

    features = np.mean(lfccs, axis=0)

    return features.reshape(1, -1)


# ==========================================
# 3. PROFESSIONAL RESULT DESIGN
# ==========================================

def create_result_card(
    prediction,
    real_prob,
    fake_prob
):

    if prediction == 0:

        status = "AI-GENERATED VOICE"
        subtitle = "Potential synthetic audio detected"
        color = "#fb7185"
        icon = "⚠"
        confidence = fake_prob

    else:

        status = "REAL HUMAN VOICE"
        subtitle = "Human voice characteristics detected"
        color = "#34d399"
        icon = "✓"
        confidence = real_prob

    return f"""
    <div class="result-card">

        <div class="result-top">
            <span class="result-label">
                ANALYSIS COMPLETE
            </span>

            <span class="result-indicator">
                ● SYSTEM ACTIVE
            </span>
        </div>

        <div class="result-icon"
             style="color:{color};">
            {icon}
        </div>

        <h2 style="color:{color};">
            {status}
        </h2>

        <p class="result-subtitle">
            {subtitle}
        </p>

        <div class="confidence-box">

            <div class="confidence-heading">
                <span>Prediction Score</span>
                <strong>{confidence:.2f}%</strong>
            </div>

            <div class="progress-track">
                <div class="progress-fill"
                     style="width:{confidence:.2f}%;
                            background:{color};">
                </div>
            </div>

        </div>

        <div class="probability-section">

            <div class="probability-item">
                <div class="probability-heading">
                    <span>Real Human Voice</span>
                    <strong>{real_prob:.2f}%</strong>
                </div>

                <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{real_prob:.2f}%;
                                background:#34d399;">
                    </div>
                </div>
            </div>

            <div class="probability-item">
                <div class="probability-heading">
                    <span>AI-Generated Voice</span>
                    <strong>{fake_prob:.2f}%</strong>
                </div>

                <div class="progress-track">
                    <div class="progress-fill"
                         style="width:{fake_prob:.2f}%;
                                background:#fb7185;">
                    </div>
                </div>
            </div>

        </div>

        <div class="result-note">
            <strong>Important:</strong>
            This result is an ML model prediction,
            not definitive proof of audio authenticity.
        </div>

    </div>
    """


# ==========================================
# 4. AUDIO ANALYSIS
# ==========================================

def analyze_audio(audio_file_path):

    if audio_file_path is None:

        return """
        <div class="empty-result">
            <h2>Waiting for Audio</h2>
            <p>Upload or record an audio sample to begin analysis.</p>
        </div>
        """

    try:

        features = extract_audio_features(
            audio_file_path
        )

        # Match training feature names

        if hasattr(model, "feature_names_in_"):

            feature_names = model.feature_names_in_

            features_df = pd.DataFrame(
                features,
                columns=feature_names
            )

        else:
            features_df = features

        # Check feature compatibility

        if hasattr(model, "n_features_in_"):

            if features.shape[1] != model.n_features_in_:
                raise ValueError(
                    "Feature mismatch between training and app."
                )

        # Prediction

        prediction = int(
            model.predict(features_df)[0]
        )

        probabilities = model.predict_proba(
            features_df
        )[0]

        # Get class probabilities

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        fake_prob = float(
            class_probabilities.get(0, 0)
        ) * 100

        real_prob = float(
            class_probabilities.get(1, 0)
        ) * 100

        if prediction not in [0, 1]:
            raise ValueError("Unknown prediction label.")

        return create_result_card(
            prediction,
            real_prob,
            fake_prob
        )

    except Exception as e:

        error_message = html.escape(str(e))

        return f"""
        <div class="error-card">
            <h3>Analysis Error</h3>
            <p>{error_message}</p>
            <p>Please check your audio file and try again.</p>
        </div>
        """


# ==========================================
# 5. PROFESSIONAL CUSTOM CSS
# ==========================================

custom_css = """

/* Main Background */

.gradio-container {
    background: #080d1b !important;
    max-width: 1250px !important;
    margin: auto !important;
    font-family: 'Inter', sans-serif !important;
}

/* Main Header */

.hero {
    background: linear-gradient(
        135deg,
        #111c35,
        #10172a,
        #0c2033
    );

    border: 1px solid #243653;
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 43px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #38bdf8;
    margin-bottom: 8px;
}

.hero p {
    color: #94a3b8;
    font-size: 15px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    background: #123047;
    color: #38bdf8;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 12px;
}

/* Section Headings */

.section-heading {
    color: #e2e8f0;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Input and Output Panels */

.panel {
    background: #111a2d;
    border: 1px solid #25344d;
    border-radius: 16px;
    padding: 22px;
    min-height: 390px;
}

/* Buttons */

.primary-btn {
    background: linear-gradient(
        90deg,
        #0284c7,
        #2563eb
    ) !important;

    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 700 !important;
    height: 50px !important;
    transition: 0.3s !important;
}

.primary-btn:hover {
    background: linear-gradient(
        90deg,
        #0ea5e9,
        #3b82f6
    ) !important;
}

/* Empty Result */

.empty-result {
    background: #111a2d;
    border: 1px dashed #334155;
    border-radius: 15px;
    padding: 65px 15px;
    text-align: center;
    color: #94a3b8;
    margin-top: 10px;
}

.empty-result h2 {
    color: #38bdf8;
}

/* Result Card */

.result-card {
    background: #111a2d;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 22px;
    margin-top: 10px;
}

.result-top {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
}

.result-label {
    color: #38bdf8;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}

.result-indicator {
    color: #34d399;
    font-size: 11px;
}

.result-icon {
    font-size: 35px;
    margin-top: 25px;
}

.result-card h2 {
    font-size: 22px;
    font-weight: 800;
    margin: 8px 0;
}

.result-subtitle {
    color: #94a3b8;
    font-size: 13px;
}

/* Confidence */

.confidence-box {
    background: #18243a;
    padding: 17px;
    border-radius: 12px;
    margin-top: 22px;
}

.confidence-heading,
.probability-heading {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: #cbd5e1;
    font-size: 13px;
    margin-bottom: 10px;
}

.confidence-heading strong {
    color: #38bdf8;
    font-size: 19px;
}

/* Progress Bars */

.progress-track {
    height: 9px;
    background: #263449;
    border-radius: 20px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.5s ease;
}

.probability-section {
    margin-top: 22px;
}

.probability-item {
    margin-bottom: 20px;
}

/* Warning */

.result-note {
    background: #1d2a3e;
    color: #94a3b8;
    border-left: 3px solid #38bdf8;
    padding: 12px;
    border-radius: 7px;
    font-size: 12px;
    margin-top: 25px;
}

.result-note strong {
    color: #38bdf8;
}

/* Error */

.error-card {
    background: #351923;
    border: 1px solid #fb7185;
    padding: 20px;
    border-radius: 12px;
    color: #fecdd3;
}

/* Footer */

.footer {
    text-align: center;
    padding: 25px;
    color: #64748b;
    font-size: 12px;
}

.footer span {
    color: #38bdf8;
}

/* Responsive */

@media(max-width: 768px) {

    .hero {
        padding: 22px;
    }

    .hero h1 {
        font-size: 30px;
    }

    .panel {
        min-height: auto;
    }

}

"""


# ==========================================
# 6. PROFESSIONAL GRADIO DASHBOARD
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="EchoGuard AI | Voice Detection"
) as demo:

    # Header

    gr.HTML("""
        <div class="hero">

            <div class="hero-badge">
                AI POWERED AUDIO SECURITY
            </div>

            <h1>◈ EchoGuard AI</h1>

            <p>
                Intelligent Voice Authenticity Detection Platform
            </p>

            <p>
                Detect potential AI-generated voices using
                machine learning and LFCC audio features.
            </p>

        </div>
    """)

    # Main Dashboard

    with gr.Row(equal_height=True):

        # Left Panel

        with gr.Column(
            scale=1,
            elem_classes=["panel"]
        ):

            gr.HTML("""
                <div class="section-heading">
                    01 / AUDIO INPUT
                </div>

                <p style="color:#94a3b8;font-size:13px;">
                    Upload an audio recording or record your
                    voice directly for authenticity analysis.
                </p>
            """)

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Upload / Recording"
            )

            submit_btn = gr.Button(
                "◈  SCAN VOICE AUTHENTICITY",
                variant="primary",
                elem_classes=["primary-btn"]
            )

            gr.HTML("""
                <div style="
                    margin-top:18px;
                    color:#64748b;
                    font-size:12px;
                    line-height:1.8;
                ">
                    <b style="color:#38bdf8;">
                        SUPPORTED FORMATS
                    </b><br>
                    WAV · MP3 · FLAC<br><br>

                    <b style="color:#38bdf8;">
                        ANALYSIS ENGINE
                    </b><br>
                    LFCC + Random Forest Classifier
                </div>
            """)

        # Right Panel

        with gr.Column(
            scale=1,
            elem_classes=["panel"]
        ):

            gr.HTML("""
                <div class="section-heading">
                    02 / DETECTION RESULTS
                </div>
            """)

            result_status = gr.HTML(
                value="""
                <div class="empty-result">
                    <div style="font-size:35px;">
                        ◈
                    </div>

                    <h2>System Ready</h2>

                    <p>
                        Your analysis results will appear here
                        after scanning the audio.
                    </p>
                </div>
                """
            )

    # How it works

    gr.HTML("""
        <div style="
            margin-top:35px;
            padding:25px;
            background:#111a2d;
            border:1px solid #25344d;
            border-radius:15px;
        ">

            <h2 style="
                color:#38bdf8;
                text-align:center;
                font-size:21px;
            ">
                HOW ECHOGUARD AI WORKS
            </h2>

            <div style="
                display:flex;
                justify-content:space-around;
                flex-wrap:wrap;
                gap:20px;
                margin-top:25px;
                text-align:center;
            ">

                <div>
                    <h2 style="color:#38bdf8;">01</h2>
                    <b style="color:#e2e8f0;">
                        Audio Input
                    </b>
                    <p style="color:#94a3b8;font-size:12px;">
                        Upload or record audio
                    </p>
                </div>

                <div>
                    <h2 style="color:#38bdf8;">02</h2>
                    <b style="color:#e2e8f0;">
                        LFCC Extraction
                    </b>
                    <p style="color:#94a3b8;font-size:12px;">
                        Extract audio features
                    </p>
                </div>

                <div>
                    <h2 style="color:#38bdf8;">03</h2>
                    <b style="color:#e2e8f0;">
                        AI Classification
                    </b>
                    <p style="color:#94a3b8;font-size:12px;">
                        Random Forest prediction
                    </p>
                </div>

                <div>
                    <h2 style="color:#38bdf8;">04</h2>
                    <b style="color:#e2e8f0;">
                        Final Results
                    </b>
                    <p style="color:#94a3b8;font-size:12px;">
                        Real / AI-generated scores
                    </p>
                </div>

            </div>
        </div>
    """)

    # Footer

    gr.HTML("""
        <div class="footer">
            <span>◈ EchoGuard AI</span>
            <br>
            AI-Based Voice Authenticity Detection System
            <br>
            Built with Python · Gradio · Scikit-learn · Librosa
        </div>
    """)

    # Button Event

    submit_btn.click(
        fn=analyze_audio,
        inputs=audio_input,
        outputs=result_status
    )


# ==========================================
# 7. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=True
    )
