import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import whisper
import yake
from dotenv import load_dotenv
from google import genai
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from textblob import TextBlob

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Project" / "dataset"
HISTORY_FILE = BASE_DIR / "meeting_history.csv"
MODEL_COMPARISON_FILE = BASE_DIR / "model_comparison.csv"
ASSETS_DIR = BASE_DIR / "assets"

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to the .env file as "
        "GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
}

# Models are loaded lazily so the page can open before Whisper finishes loading.
_whisper_model = None
kw_extractor = yake.KeywordExtractor(top=5)


# =========================================================
# MODEL AND UTILITY FUNCTIONS
# =========================================================

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def load_custom_css() -> str:
    css_file = ASSETS_DIR / "style.css"
    if css_file.exists():
        return css_file.read_text(encoding="utf-8")
    return ""


def extract_action_items(transcript: str) -> list[str]:
    action_items = []
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
        "deliver",
    ]

    for sentence in transcript.split("."):
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue

        if any(word in cleaned_sentence.lower() for word in action_words):
            action_items.append(cleaned_sentence)

    return action_items


def safe_paragraph_text(text: str) -> str:
    """Escape characters that ReportLab Paragraph treats as markup."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def create_pdf(
    meeting_title: str,
    transcript: str,
    summary: str,
    sentiment: str,
    keywords: list[str],
    action_items: list[str],
) -> str:
    output_dir = Path(tempfile.mkdtemp(prefix="meeting_report_"))
    pdf_path = output_dir / "Meeting_Report.pdf"

    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>AI Meeting Intelligence Report</b>", styles["Title"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph(
            f"<b>Meeting Title:</b> {safe_paragraph_text(meeting_title)}",
            styles["BodyText"],
        ),
        Paragraph(
            f"<b>Sentiment:</b> {safe_paragraph_text(sentiment)}",
            styles["BodyText"],
        ),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph("<b>Summary</b>", styles["Heading2"]),
        Paragraph(safe_paragraph_text(summary), styles["BodyText"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph("<b>Transcript</b>", styles["Heading2"]),
        Paragraph(safe_paragraph_text(transcript), styles["BodyText"]),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph("<b>Keywords</b>", styles["Heading2"]),
        Paragraph(
            safe_paragraph_text(", ".join(keywords)),
            styles["BodyText"],
        ),
        Paragraph("<br/>", styles["Normal"]),
        Paragraph("<b>Action Items</b>", styles["Heading2"]),
    ]

    if action_items:
        for item in action_items:
            story.append(
                Paragraph(
                    "• " + safe_paragraph_text(item),
                    styles["BodyText"],
                )
            )
    else:
        story.append(Paragraph("No Action Items Found", styles["BodyText"]))

    doc.build(story)
    return str(pdf_path)


def create_transcript_file(transcript: str) -> str:
    output_dir = Path(tempfile.mkdtemp(prefix="meeting_transcript_"))
    transcript_path = output_dir / "meeting_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    return str(transcript_path)


def format_keywords(keywords: list[str]) -> str:
    if not keywords:
        return "No keywords found."
    return "\n".join(f"- ✅ {keyword}" for keyword in keywords)


def format_action_items(action_items: list[str]) -> str:
    if not action_items:
        return "No action items found."
    return "\n".join(f"- {item}" for item in action_items)


def save_meeting_history(
    meeting_title: str,
    transcript: str,
    sentiment: str,
    keywords: list[str],
    processing_time: float,
) -> None:
    new_meeting = pd.DataFrame(
        {
            "Date": [datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
            "Meeting Title": [meeting_title.strip()],
            "Transcript": [transcript],
            "Sentiment": [sentiment],
            "Keywords": [", ".join(keywords)],
            "Processing Time": [processing_time],
        }
    )

    if HISTORY_FILE.exists():
        try:
            old_history = pd.read_csv(HISTORY_FILE)
            new_meeting = pd.concat(
                [old_history, new_meeting],
                ignore_index=True,
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    new_meeting.to_csv(HISTORY_FILE, index=False)


# =========================================================
# MAIN ANALYSIS
# =========================================================

def analyze_meeting(audio_path: str | None, language: str):
    if not audio_path:
        raise gr.Error("Please upload a meeting audio file first.")

    start_time = time.time()

    try:
        whisper_model = get_whisper_model()

        result = whisper_model.transcribe(
            audio_path,
            language="en",
        )
        transcript = result.get("text", "").strip()

        if not transcript:
            raise gr.Error("No speech could be detected in the uploaded audio.")

        summary_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "Summarize this meeting clearly. Include the main discussion, "
                "important decisions, and key outcomes.\n\n"
                f"Transcript:\n{transcript}"
            ),
        )
        summary = (summary_response.text or "").strip()

        title_response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "Generate only a short meeting title with a maximum of 8 words. "
                "Do not add quotation marks or an explanation.\n\n"
                f"Transcript:\n{transcript}"
            ),
        )
        meeting_title = (title_response.text or "Untitled Meeting").strip()

        selected_code = LANGUAGE_CODES.get(language, "en")
        if selected_code != "en":
            from deep_translator import GoogleTranslator

            summary = GoogleTranslator(
                source="auto",
                target=selected_code,
            ).translate(summary)

        polarity = TextBlob(transcript).sentiment.polarity
        if polarity > 0:
            sentiment = "😊 Positive"
        elif polarity < 0:
            sentiment = "😟 Negative"
        else:
            sentiment = "😐 Neutral"

        keyword_pairs = kw_extractor.extract_keywords(transcript)
        keywords = [item[0] for item in keyword_pairs]
        action_items = extract_action_items(transcript)

        processing_time = round(time.time() - start_time, 2)

        transcript_file = create_transcript_file(transcript)
        pdf_file = create_pdf(
            meeting_title=meeting_title,
            transcript=transcript,
            summary=summary,
            sentiment=sentiment,
            keywords=keywords,
            action_items=action_items,
        )

        save_meeting_history(
            meeting_title=meeting_title,
            transcript=transcript,
            sentiment=sentiment,
            keywords=keywords,
            processing_time=processing_time,
        )

        status = (
            f"✅ Meeting analysis completed and saved in "
            f"{processing_time} seconds."
        )

        return (
            status,
            processing_time,
            transcript,
            summary,
            meeting_title,
            sentiment,
            format_keywords(keywords),
            format_action_items(action_items),
            transcript_file,
            pdf_file,
        )

    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Meeting analysis failed: {error}") from error


# =========================================================
# PAGE DATA FUNCTIONS
# =========================================================

def load_home_statistics():
    files = {
        "Train": DATASET_DIR / "train.json",
        "Validation": DATASET_DIR / "validation.json",
        "Test": DATASET_DIR / "test.json",
    }

    counts = {}
    missing_files = []

    for name, file_path in files.items():
        if file_path.exists():
            try:
                counts[name] = len(pd.read_json(file_path, lines=True))
            except ValueError:
                counts[name] = 0
                missing_files.append(f"{name}: invalid JSON")
        else:
            counts[name] = 0
            missing_files.append(str(file_path.relative_to(BASE_DIR)))

    total = sum(counts.values())

    if missing_files:
        status = "⚠️ Missing or invalid dataset files: " + ", ".join(missing_files)
    else:
        status = "✅ AI Meeting Intelligence Platform is ready."

    return (
        counts["Train"],
        counts["Validation"],
        counts["Test"],
        total,
        status,
    )


def load_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame(
            columns=[
                "Date",
                "Meeting Title",
                "Transcript",
                "Sentiment",
                "Keywords",
                "Processing Time",
            ]
        )

    try:
        return pd.read_csv(HISTORY_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def build_dashboard():
    history = load_history()

    if history.empty or "Sentiment" not in history.columns:
        empty_figure_1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.text(
            0.5,
            0.5,
            "No meeting history found",
            ha="center",
            va="center",
        )
        ax1.axis("off")

        empty_figure_2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.text(
            0.5,
            0.5,
            "No processing-time data found",
            ha="center",
            va="center",
        )
        ax2.axis("off")

        return 0, 0, 0, 0, empty_figure_1, empty_figure_2

    sentiment_text = history["Sentiment"].astype(str)
    positive = sentiment_text.str.contains("Positive", na=False).sum()
    negative = sentiment_text.str.contains("Negative", na=False).sum()
    neutral = sentiment_text.str.contains("Neutral", na=False).sum()

    sentiment_counts = history["Sentiment"].value_counts()
    sentiment_figure, sentiment_axis = plt.subplots(figsize=(7, 4))
    sentiment_counts.plot(kind="bar", ax=sentiment_axis)
    sentiment_axis.set_xlabel("Sentiment")
    sentiment_axis.set_ylabel("Meetings")
    sentiment_axis.set_title("Sentiment Distribution")
    sentiment_figure.tight_layout()

    processing_figure, processing_axis = plt.subplots(figsize=(7, 4))
    if "Processing Time" in history.columns:
        processing_values = pd.to_numeric(
            history["Processing Time"],
            errors="coerce",
        )
        processing_values.plot(
            kind="line",
            marker="o",
            ax=processing_axis,
        )
    processing_axis.set_xlabel("Meeting Number")
    processing_axis.set_ylabel("Seconds")
    processing_axis.set_title("Processing Time")
    processing_figure.tight_layout()

    return (
        len(history),
        int(positive),
        int(negative),
        int(neutral),
        sentiment_figure,
        processing_figure,
    )


def load_model_comparison():
    if not MODEL_COMPARISON_FILE.exists():
        empty = pd.DataFrame(
            columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        )
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.text(0.5, 0.5, "model_comparison.csv not found", ha="center", va="center")
        ax1.axis("off")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.text(0.5, 0.5, "model_comparison.csv not found", ha="center", va="center")
        ax2.axis("off")

        return empty, "No model comparison data found.", fig1, fig2

    comparison = pd.read_csv(MODEL_COMPARISON_FILE)

    required_columns = {
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    }
    missing_columns = required_columns.difference(comparison.columns)

    if missing_columns:
        raise gr.Error(
            "model_comparison.csv is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    accuracy_figure, accuracy_axis = plt.subplots(figsize=(8, 5))
    accuracy_axis.bar(comparison["Model"], comparison["Accuracy"])
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_title("Accuracy Comparison")
    accuracy_axis.tick_params(axis="x", rotation=20)
    accuracy_figure.tight_layout()

    f1_figure, f1_axis = plt.subplots(figsize=(8, 5))
    f1_axis.bar(comparison["Model"], comparison["F1 Score"])
    f1_axis.set_ylabel("F1 Score")
    f1_axis.set_title("F1 Score Comparison")
    f1_axis.tick_params(axis="x", rotation=20)
    f1_figure.tight_layout()

    best_row = comparison.loc[comparison["Accuracy"].idxmax()]
    best_text = (
        f"🏆 Best Model: **{best_row['Model']}** "
        f"(Accuracy: **{best_row['Accuracy']:.4f}**)"
    )

    return comparison, best_text, accuracy_figure, f1_figure


# =========================================================
# GRADIO USER INTERFACE
# =========================================================

custom_css = load_custom_css()

with gr.Blocks(
    title="AI Meeting Intelligence Platform",
    css=custom_css,
) as demo:
    gr.Markdown(
        """
        # 🤖 AI Meeting Intelligence Platform
        Welcome to the AI-powered meeting analysis platform.
        """
    )

    with gr.Tabs():
        # ---------------- HOME ----------------
        with gr.Tab("🏠 Home"):
            gr.Markdown(
                """
                ## AI-Powered Meeting Assistant

                This application uses artificial intelligence to analyze
                meeting recordings.

                ### Features

                - ✅ Speech-to-text using Whisper
                - ✅ AI meeting summary
                - ✅ Sentiment analysis
                - ✅ Keyword extraction
                - ✅ Action-item detection
                - ✅ Multi-language summary
                - ✅ PDF report generation
                - ✅ Meeting history
                - ✅ Analytics dashboard
                - ✅ Machine-learning model comparison
                """
            )

            home_refresh = gr.Button("🔄 Load Dataset Statistics")

            with gr.Row():
                train_count = gr.Number(label="Train", interactive=False)
                validation_count = gr.Number(
                    label="Validation",
                    interactive=False,
                )
                test_count = gr.Number(label="Test", interactive=False)
                total_count = gr.Number(label="Total", interactive=False)

            home_status = gr.Markdown()

            home_refresh.click(
                fn=load_home_statistics,
                outputs=[
                    train_count,
                    validation_count,
                    test_count,
                    total_count,
                    home_status,
                ],
            )

        # ---------------- MEETING ANALYSIS ----------------
        with gr.Tab("🎤 Meeting Analysis"):
            with gr.Row():
                language = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value="English",
                    label="🌐 Select Summary Language",
                )
                audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload Meeting Audio",
                    format="wav",
                )

            analyze_button = gr.Button(
                "🚀 Analyze Meeting",
                variant="primary",
            )
            analysis_status = gr.Markdown()

            with gr.Row():
                processing_time = gr.Number(
                    label="Processing Time (seconds)",
                    interactive=False,
                )
                meeting_title = gr.Textbox(
                    label="📌 Meeting Title",
                    interactive=False,
                )
                sentiment = gr.Textbox(
                    label="😊 Meeting Sentiment",
                    interactive=False,
                )

            transcript = gr.Textbox(
                label="📝 Transcript",
                lines=12,
                interactive=False,
            )
            summary = gr.Textbox(
                label="📝 AI Meeting Summary",
                lines=8,
                interactive=False,
            )

            with gr.Row():
                keywords = gr.Markdown(label="🔑 Keywords")
                action_items = gr.Markdown(label="📌 Action Items")

            with gr.Row():
                transcript_download = gr.File(
                    label="📥 Download Transcript",
                    interactive=False,
                )
                pdf_download = gr.File(
                    label="📄 Download PDF Report",
                    interactive=False,
                )

            analyze_button.click(
                fn=analyze_meeting,
                inputs=[audio_input, language],
                outputs=[
                    analysis_status,
                    processing_time,
                    transcript,
                    summary,
                    meeting_title,
                    sentiment,
                    keywords,
                    action_items,
                    transcript_download,
                    pdf_download,
                ],
                show_progress="full",
            )

        # ---------------- DASHBOARD ----------------
        with gr.Tab("📊 Dashboard"):
            dashboard_refresh = gr.Button("🔄 Refresh Dashboard")

            with gr.Row():
                total_meetings = gr.Number(
                    label="Total Meetings",
                    interactive=False,
                )
                positive_meetings = gr.Number(
                    label="😊 Positive",
                    interactive=False,
                )
                negative_meetings = gr.Number(
                    label="😟 Negative",
                    interactive=False,
                )
                neutral_meetings = gr.Number(
                    label="😐 Neutral",
                    interactive=False,
                )

            with gr.Row():
                sentiment_plot = gr.Plot(
                    label="📊 Sentiment Distribution"
                )
                processing_plot = gr.Plot(
                    label="⏱ Processing Time"
                )

            dashboard_refresh.click(
                fn=build_dashboard,
                outputs=[
                    total_meetings,
                    positive_meetings,
                    negative_meetings,
                    neutral_meetings,
                    sentiment_plot,
                    processing_plot,
                ],
            )

        # ---------------- HISTORY ----------------
        with gr.Tab("📚 Meeting History"):
            history_refresh = gr.Button("🔄 Refresh Meeting History")
            history_table = gr.Dataframe(
                value=load_history,
                label="Meeting History",
                interactive=False,
                wrap=True,
            )
            history_refresh.click(
                fn=load_history,
                outputs=history_table,
            )

        # ---------------- MODEL COMPARISON ----------------
        with gr.Tab("🤖 Model Comparison"):
            comparison_refresh = gr.Button(
                "🔄 Load Model Comparison"
            )
            comparison_table = gr.Dataframe(
                label="📋 Model Comparison Table",
                interactive=False,
            )
            best_model = gr.Markdown()
            accuracy_plot = gr.Plot(label="📈 Accuracy Comparison")
            f1_plot = gr.Plot(label="📊 F1 Score Comparison")

            comparison_refresh.click(
                fn=load_model_comparison,
                outputs=[
                    comparison_table,
                    best_model,
                    accuracy_plot,
                    f1_plot,
                ],
            )

        # ---------------- ABOUT ----------------
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                # AI Meeting Intelligence Platform

                This project was developed using artificial intelligence
                and machine learning.

                ## Technologies Used

                - Gradio
                - Whisper
                - Google Gemini
                - TextBlob
                - YAKE
                - Deep Translator
                - ReportLab
                - Pandas
                - Matplotlib

                ## Features

                - Speech-to-text
                - AI meeting summary
                - Sentiment analysis
                - Keyword extraction
                - Action-item detection
                - PDF report generation
                - Meeting history
                - Dashboard
                - Model comparison

                ### Developed By

                Nisarg Patil
                """
            )

    demo.load(
        fn=load_home_statistics,
        outputs=[
            train_count,
            validation_count,
            test_count,
            total_count,
            home_status,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
    )