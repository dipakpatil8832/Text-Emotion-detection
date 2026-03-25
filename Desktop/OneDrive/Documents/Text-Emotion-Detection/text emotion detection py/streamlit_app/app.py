import streamlit as st

import pandas as pd
import numpy as np
import altair as alt

import joblib
toxic_model = joblib.load("C:/Users/dp431/Desktop/OneDrive/Documents/Text-Emotion-Detection/text emotion detection py/model/toxic_model.pkl")
toxic_vectorizer = joblib.load("C:/Users/dp431/Desktop/OneDrive/Documents/Text-Emotion-Detection/text emotion detection py/model/tfidf_vectorizer.pkl")

pipe_lr = joblib.load(open("C:/Users/dp431/Desktop/OneDrive/Documents/Text-Emotion-Detection/text emotion detection py/model/text_emotion.pkl", "rb"))


emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def predict_toxicity(text):
    text_vec = toxic_vectorizer.transform([text])
    pred = toxic_model.predict(text_vec)[0]
    prob = toxic_model.predict_proba(text_vec)[0][1]
    
    label = "Toxic" if pred == 1 else "Non-Toxic"
    return label, float(prob)



def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        toxicity, toxic_prob = predict_toxicity(raw_text)


        st.subheader("Toxicity Result")
        st.write(f"Prediction: {toxicity}")
        st.write(f"Confidence: {toxic_prob:.2f}")


        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)






if __name__ == '__main__':
    main()