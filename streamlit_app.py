import streamlit as st

st.set_page_config(
    page_title="AI Meeting Intelligence Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Meeting Intelligence Platform")

st.write("Welcome to the AI Meeting Intelligence Platform")

import whisper
import pandas as pd
import tempfile
import time
import os
import csv
import matplotlib.pyplot as plt
from google import genai

from datetime import datetime
from transformers import pipeline
from deep_translator import GoogleTranslator
from textblob import TextBlob
import yake

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

def load_css():
    if os.path.exists("assets/style.css"):
        with open("assets/style.css") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )



model = load_whisper()
summarizer = load_summarizer()

kw_extractor = yake.KeywordExtractor(top=5)

def extract_action_items(transcript):

    action_items = []

    sentences = transcript.split(".")

    action_words = [
        "will",
        "should",
        "must",
        "need to",
        "assign",
        "complete",
        "finish",
        "submit",
        "send",
        "prepare",
        "review",
        "update",
        "schedule",
        "deliver"
    ]

    for sentence in sentences:
        for word in action_words:
            if word.lower() in sentence.lower():
                action_items.append(sentence.strip())
                break

    return action_items

def create_pdf(meeting_title, transcript, summary, sentiment, keywords, action_items):

    pdf_path = "Meeting_Report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>AI Meeting Intelligence Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph(f"<b>Meeting Title:</b> {meeting_title}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Sentiment:</b> {sentiment}", styles["BodyText"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(Paragraph(summary, styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Transcript</b>", styles["Heading2"]))
    story.append(Paragraph(transcript, styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Keywords</b>", styles["Heading2"]))
    story.append(Paragraph(", ".join(keywords), styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Action Items</b>", styles["Heading2"]))

    if action_items:
        for item in action_items:
            story.append(Paragraph("• " + item, styles["BodyText"]))
    else:
        story.append(Paragraph("No Action Items Found", styles["BodyText"]))

    doc.build(story)

    return pdf_path

# ==========================
# Sidebar Navigation
# ==========================

st.sidebar.title("🤖 AI Meeting Intelligence")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎤 Meeting Analysis",
        "📊 Dashboard",
        "📚 Meeting History",
        "🤖 Model Comparison",
        "ℹ️ About"
    ]
)

# ==========================
# Home Page
# ==========================

if menu == "🏠 Home":

    st.title("🤖 AI Meeting Intelligence Platform")

    st.markdown("""
    ## AI Powered Meeting Assistant

    This application uses Artificial Intelligence to analyze meeting recordings.

    ### Features

    ✅ Speech-to-Text using Whisper

    ✅ AI Meeting Summary

    ✅ Sentiment Analysis

    ✅ Keyword Extraction

    ✅ Action Item Detection

    ✅ Multi-language Summary

    ✅ PDF Report Generation

    ✅ Meeting History

    ✅ Dashboard

    ✅ Machine Learning Model Comparison
    """)

    train = pd.read_json("Project/dataset/train.json", lines=True)
    validation = pd.read_json("Project/dataset/validation.json", lines=True)
    test = pd.read_json("Project/dataset/test.json", lines=True)

    total_records = len(train) + len(validation) + len(test)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Train", len(train))
    col2.metric("Validation", len(validation))
    col3.metric("Test", len(test))
    col4.metric("Total", total_records)

    st.divider()

    st.success("✔ AI Meeting Intelligence Platform is Ready")

# ==========================
# Meeting Analysis
# ==========================

if menu == "🎤 Meeting Analysis":

    st.title("🎤 Meeting Analysis")

    language = st.selectbox(
        "🌐 Select Summary Language",
        [
            "English",
            "Hindi",
            "Marathi",
            "French",
            "German",
            "Spanish"
        ]
    )

    language_code = {
        "English": "en",
        "Hindi": "hi",
        "Marathi": "mr",
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }[language]

    uploaded_file = st.file_uploader(
        "Upload Meeting Audio",
        type=["mp3", "wav", "m4a"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        if st.button("🚀 Analyze Meeting"):

            start = time.time()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

                tmp.write(uploaded_file.read())

                audio_path = tmp.name

            with st.spinner("Transcribing audio..."):

                result = model.transcribe(audio_path, language="en")

                transcript = result["text"]

            processing_time = round(time.time() - start, 2)

            st.success("✅ Transcription Completed")

            st.metric(
                "Processing Time (seconds)",
                processing_time
            )

            st.subheader("📝 Transcript")

            st.write(transcript)

            st.download_button(
    "📥 Download Transcript",
    transcript,
    file_name="meeting_transcript.txt",
    key="download_transcript"
)
                      # ==========================
            # AI Meeting Summary
            # ==========================

            st.subheader("📝 AI Meeting Summary")

            with st.spinner("Generating Summary..."):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=f"Summarize this meeting:\n\n{transcript}"
                )

                summary = response.text

            st.write(summary)

            # ==========================
            # Meeting Title
            # ==========================

            st.subheader("📌 Meeting Title")

            title_response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=f"""
Generate a short meeting title (maximum 8 words).

Transcript:
{transcript}
"""
            )

            meeting_title = title_response.text

            st.info(meeting_title)

            # ==========================
            # Sentiment Analysis
            # ==========================

            analysis = TextBlob(transcript)

            polarity = analysis.sentiment.polarity

            if polarity > 0:
                sentiment = "😊 Positive"
            elif polarity < 0:
                sentiment = "😟 Negative"
            else:
                sentiment = "😐 Neutral"

            st.subheader("😊 Meeting Sentiment")
            st.success(sentiment)

            # ==========================
            # Keywords
            # ==========================

            keywords = kw_extractor.extract_keywords(transcript)
            keywords = [item[0] for item in keywords]

            st.subheader("🔑 Keywords")

            for word in keywords:
                st.write("✅", word)

            # ==========================
            # Action Items
            # ==========================

            action_items = extract_action_items(transcript)

            st.subheader("📌 Action Items")

            if len(action_items) == 0:
                st.info("No Action Items Found")
            else:
                for item in action_items:
                    st.success(item)

            # ==========================
            # PDF Report
            # ==========================

            pdf_file = create_pdf(
                meeting_title,
                transcript,
                summary,
                sentiment,
                keywords,
                action_items
            )

            with open(pdf_file, "rb") as file:
                st.download_button(
                    "📄 Download PDF Report",
                    file,
                    file_name="Meeting_Report.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )

            # ==========================
            # Save Meeting History
            # ==========================

            history_file = "meeting_history.csv"

            meeting = pd.DataFrame({
                "Date": [datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
                "Meeting Title": [meeting_title],
                "Transcript": [transcript],
                "Sentiment": [sentiment],
                "Keywords": [", ".join(keywords)],
                "Processing Time": [processing_time]
            })

            if os.path.exists(history_file):
                old_history = pd.read_csv(history_file)
                meeting = pd.concat([old_history, meeting], ignore_index=True)

            meeting.to_csv(history_file, index=False)

            st.success("✅ Meeting saved successfully!")
# ==========================
# Meeting History
# ==========================

if menu == "📚 Meeting History":

    st.title("📚 Meeting History")

    if os.path.exists("meeting_history.csv"):

        history = pd.read_csv("meeting_history.csv")

        st.dataframe(history, use_container_width=True)

    else:

        st.warning("No meeting history found.")

if menu == "📊 Dashboard":

    st.title("📊 AI Meeting Dashboard")

    if os.path.exists("meeting_history.csv"):

        history = pd.read_csv("meeting_history.csv")

        st.subheader("Meeting Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Meetings", len(history))

        positive = len(
            history[
                history["Sentiment"].astype(str).str.contains(
                    "Positive",
                    na=False
                )
            ]
        )

        negative = len(
            history[
                history["Sentiment"].astype(str).str.contains(
                    "Negative",
                    na=False
                )
            ]
        )

        neutral = len(
            history[
                history["Sentiment"].astype(str).str.contains(
                    "Neutral",
                    na=False
                )
            ]
        )

        col2.metric("😊 Positive", positive)
        col3.metric("😟 Negative", negative)
        col4.metric("😐 Neutral", neutral)

        st.divider()

        st.subheader("📊 Sentiment Distribution")

        sentiment_counts = history["Sentiment"].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))

        sentiment_counts.plot(kind="bar", ax=ax)

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Meetings")

        st.pyplot(fig)

        st.divider()

        st.subheader("⏱ Processing Time")

        fig2, ax2 = plt.subplots(figsize=(7,4))

        history["Processing Time"].plot(
            kind="line",
            marker="o",
            ax=ax2
        )

        ax2.set_ylabel("Seconds")

        st.pyplot(fig2)

    else:

        st.warning("No meeting history found.")

# ==========================
# Model Comparison
# ==========================

if menu == "🤖 Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    if os.path.exists("model_comparison.csv"):

        comparison = pd.read_csv("model_comparison.csv")

        st.subheader("📋 Model Comparison Table")
        st.dataframe(comparison, use_container_width=True)

        st.subheader("📌 Individual Model Performance")

        for _, row in comparison.iterrows():

            with st.expander(f"🤖 {row['Model']}"):

                col1, col2 = st.columns(2)

                col1.metric("Accuracy", f"{row['Accuracy']:.4f}")
                col2.metric("Precision", f"{row['Precision']:.4f}")

                col1.metric("Recall", f"{row['Recall']:.4f}")
                col2.metric("F1 Score", f"{row['F1 Score']:.4f}")

        st.subheader("📈 Accuracy Comparison")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(comparison["Model"], comparison["Accuracy"])
        ax.set_ylabel("Accuracy")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        st.subheader("📊 F1 Score Comparison")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(comparison["Model"], comparison["F1 Score"])
        ax2.set_ylabel("F1 Score")
        plt.xticks(rotation=20)
        st.pyplot(fig2)

        best_model = comparison.loc[
            comparison["Accuracy"].idxmax(),
            "Model"
        ]

        best_accuracy = comparison["Accuracy"].max()

        st.success(
            f"🏆 Best Model: {best_model} ({best_accuracy:.4f})"
        )

    else:
        st.warning("model_comparison.csv not found.")

# ==========================
# About
# ==========================

if menu == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("""
# AI Meeting Intelligence Platform

This project was developed using Artificial Intelligence and Machine Learning.

## Technologies Used
- Streamlit
- Whisper
- Google Gemini 2.5 Flash Lite
- TextBlob
- YAKE
- Google Translator
- ReportLab
- Pandas
- Matplotlib

## Features
- Speech-to-Text
- AI Meeting Summary
- Sentiment Analysis
- Keyword Extraction
- Action Item Detection
- PDF Report Generation
- Meeting History
- Dashboard
- Model Comparison

### Developed By
Nisarg Patil
""")