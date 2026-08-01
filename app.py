from flask import Flask, render_template, request, send_file, Response
from deep_translator import GoogleTranslator
from datetime import datetime
from transformers import pipeline
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from textblob import TextBlob
import os
import whisper
import time
import pandas as pd
import yake
import csv
from datetime import datetime
import matplotlib.pyplot as plt


train = pd.read_json("Project/dataset/train.json", lines=True)
validation = pd.read_json("Project/dataset/validation.json", lines=True)
test = pd.read_json("Project/dataset/test.json", lines=True)

train_records = len(train)
validation_records = len(validation)
test_records = len(test)

total_records = (
    train_records
    + validation_records
    + test_records
)

print("Train:", train_records)
print("Validation:", validation_records)
print("Test:", test_records)
print("Total:", total_records)

app = Flask(__name__)
last_transcript = ""
last_summary = ""
last_word_count = 0
last_char_count = 0
last_summary_words = 0
last_processing_time = 0
last_title = ""

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
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

@app.route("/", methods=["GET", "POST"])
def upload_file():

    transcript = ""
    summary = ""
    meeting_title = ""
    sentiment = "Not Analyzed"
    keywords = []
    action_items = []
    language = request.form.get("language", "en")

    word_count = 0
    char_count = 0
    summary_words = 0
    processing_time = 0

    if request.method == "POST":

        file = request.files["file"]

        if file:
            start = time.time()

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = model.transcribe(filepath, language="en")
            transcript = result["text"]

            summary = summarizer(
                transcript,
                max_length=150,
                min_length=50,
                do_sample=False
            )

            summary = summary[0]["summary_text"]
            title_prompt = summarizer(
            transcript,
            max_length=12,
            min_length=5,
            do_sample=False
    )

            meeting_title = title_prompt[0]["summary_text"]
            if language != "en":
               summary = GoogleTranslator(
               source="auto",
               target=language
            ).translate(summary) 
            keywords = kw_extractor.extract_keywords(transcript)
            keywords = [item[0] for item in keywords]
            action_items = extract_action_items(transcript)

            analysis = TextBlob(transcript)
            polarity = analysis.sentiment.polarity

            if polarity > 0:
              sentiment = "😊 Positive"
            elif polarity < 0:
              sentiment = "😟 Negative"
            else:
              sentiment = "😐 Neutral"

            word_count = len(transcript.split())
            char_count = len(transcript)
            summary_words = len(summary.split())
            processing_time = round(time.time() - start, 2)

            global last_transcript
            global last_summary
            global last_word_count
            global last_char_count
            global last_summary_words
            global last_processing_time
            global last_title 
            last_title = meeting_title

            file_exists = os.path.isfile("meeting_history.csv")

            with open("meeting_history.csv", "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                if not file_exists:
                    writer.writerow([
                        "Date",
                        "Transcript",
                        "Summary",
                        "Sentiment",
                        "Keywords",
                        "Word Count",
                        "Processing Time"
                    ])

                writer.writerow([
                    datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    transcript,
                    summary,
                    sentiment,
                    ", ".join(keywords),
                    word_count,
                    processing_time
                ])

            last_transcript = transcript
            last_summary = summary
            last_word_count = word_count
            last_char_count = char_count
            last_summary_words = summary_words
            last_processing_time = processing_time

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary,
        sentiment=sentiment,
        keywords=keywords,
        action_items=action_items,
        meeting_title=meeting_title,
        word_count=word_count,
        char_count=char_count,
        summary_words=summary_words,
        processing_time=processing_time,
        total_records=total_records,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
    )

@app.route("/download_transcript")
def download_transcript():
    return Response(
        last_transcript,
        mimetype="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=meeting_transcript.txt"
        }
    )


@app.route("/history")
def history():
    history = pd.read_csv("meeting_history.csv")

    sentiment_counts = history["Sentiment"].value_counts()

    plt.figure(figsize=(5,4))
    sentiment_counts.plot(kind="bar")
    plt.title("Meeting Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()

    image_folder = os.path.join(app.root_path, "static", "images")

    os.makedirs(image_folder, exist_ok=True)

    save_path = os.path.join(image_folder, "sentiment_chart.png")

    plt.savefig(save_path)
    plt.close()

    return render_template("history.html", history=history)

@app.route("/download_pdf")
def download_pdf():

    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph

    pdf = SimpleDocTemplate("Meeting_Report.pdf")

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>AI Meeting Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Transcript:</b>", styles["Heading2"]))
    story.append(Paragraph(last_transcript, styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["Normal"]))

    story.append(Paragraph("<b>Summary:</b>", styles["Heading2"]))
    story.append(Paragraph(last_summary, styles["BodyText"]))

    pdf.build(story)

    return send_file(
        "Meeting_Report.pdf",
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)