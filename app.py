import os
import re
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path():
    candidate_paths = [
        os.path.join(CURRENT_DIR, "model", "text_emotion.pkl"),
        os.path.join(CURRENT_DIR, "..", "model", "text_emotion.pkl"),
        os.path.join(os.getcwd(), "model", "text_emotion.pkl"),
        os.path.join(CURRENT_DIR, "text_emotion.pkl"),
        os.path.join(os.getcwd(), "text_emotion.pkl"),
    ]
    for path in candidate_paths:
        resolved = os.path.abspath(path)
        if os.path.exists(resolved):
            return resolved
    return os.path.abspath(os.path.join(CURRENT_DIR, "model", "text_emotion.pkl"))

@st.cache_resource
def load_emotion_pipeline():
    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Please run 'train_model.py' first."
        )
    return joblib.load(model_path)

def clean_input_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨",
    "happy": "🤗",
    "joy": "😄",
    "neutral": "😐",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
    "uncertain": "🤔"
}

def main():
    st.set_page_config(
        page_title="Text Emotion Detection",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        pipeline = load_emotion_pipeline()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.title("🎭 Text Emotion Detection")
    st.markdown(
        "Detect emotions in text with class balance, negation awareness, and confidence thresholding."
    )

    with st.sidebar:
        st.header("⚙️ Model Settings")
        st.markdown(
            "Configure how the model handles ambiguous, low-confidence, or physical state inputs."
        )
        confidence_threshold = st.slider(
            "Confidence Rejection Threshold",
            min_value=0.20,
            max_value=0.85,
            value=0.40,
            step=0.05,
            help="Predictions below this confidence score will be classified as 'Uncertain / Neutral' instead of forcing a low-confidence emotion label."
        )

        st.markdown("---")
        st.markdown("### 📊 Supported Emotion Classes")
        for cls_name in pipeline.classes_:
            emoji = emotions_emoji_dict.get(cls_name, "✨")
            st.write(f"- **{cls_name.capitalize()}** {emoji}")

    with st.form(key="emotion_form"):
        raw_text = st.text_area(
            "Enter text to analyze:",
            placeholder="e.g. 'i am hungry', 'I won the lottery today!', 'I am not happy with this service'",
            height=120
        )
        submit_text = st.form_submit_button(label="Analyze Emotion", use_container_width=True)

    if submit_text:
        if not raw_text.strip():
            st.warning("Please enter some text before submitting.")
            return

        cleaned_text = clean_input_text(raw_text)
        probabilities = pipeline.predict_proba([cleaned_text])[0]
        classes = pipeline.classes_

        max_idx = np.argmax(probabilities)
        max_confidence = probabilities[max_idx]
        top_predicted_emotion = classes[max_idx]

        is_below_threshold = max_confidence < confidence_threshold

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.subheader("📋 Detection Result")
            st.markdown(f"**Input Text:** *\"{raw_text}\"*")

            if is_below_threshold:
                st.warning("⚠️ **Low Confidence / Uncertain State**")
                st.info(
                    f"The top detected class is **{top_predicted_emotion}** with only **{max_confidence:.1%}** confidence, "
                    f"which is below your threshold of **{confidence_threshold:.0%}**.\n\n"
                    f"💡 *This input is likely a **physical state** (e.g. hungry, tired, cold) or **neutral context** without strong emotional expression.*"
                )
                emoji_icon = emotions_emoji_dict.get("uncertain", "🤔")
                st.metric(
                    label="Status",
                    value=f"Uncertain / Neutral {emoji_icon}",
                    delta=f"-{(confidence_threshold - max_confidence)*100:.1f}% below threshold",
                    delta_color="inverse"
                )
            else:
                emoji_icon = emotions_emoji_dict.get(top_predicted_emotion, "✨")
                st.success(f"### {top_predicted_emotion.upper()} {emoji_icon}")
                st.metric(
                    label="Confidence Score",
                    value=f"{max_confidence * 100:.2f}%",
                    delta=f"+{(max_confidence - confidence_threshold)*100:.1f}% above threshold"
                )

        with col2:
            st.subheader("📊 Probability Breakdown")
            proba_df = pd.DataFrame({
                "Emotion": [c.capitalize() for c in classes],
                "RawEmotion": classes,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            chart = (
                alt.Chart(proba_df)
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    x=alt.X(
                        "Probability:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%", title="Probability")
                    ),
                    y=alt.Y("Emotion:N", sort="-x", title="Emotion Class"),
                    color=alt.Color(
                        "Probability:Q",
                        scale=alt.Scale(scheme="tealblues"),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("Emotion:N"),
                        alt.Tooltip("Probability:Q", format=".2%")
                    ]
                )
                .properties(height=320)
            )

            st.altair_chart(chart, use_container_width=True)

            with st.expander("View Raw Probability Table"):
                display_df = proba_df[["Emotion", "Probability"]].copy()
                display_df["Probability"] = display_df["Probability"].apply(lambda p: f"{p:.2%}")
                st.dataframe(display_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()