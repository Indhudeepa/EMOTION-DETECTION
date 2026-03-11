import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sqlite3
from flask import Flask, render_template, jsonify, request, send_file
from fpdf import FPDF
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------------
# DATABASE INITIALIZATION
# -------------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# -------------------------------
# LOAD MODEL AND FACE CASCADE
# -------------------------------
model = load_model("emotion_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# -------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------
recommendations = {

    "Happy": {
        "music":[

{"name":"Vaathi Coming",
"link":"https://www.youtube.com/watch?v=0n7AWxYCj9I"},

{"name":"Rowdy Baby",
"link":"https://www.youtube.com/watch?v=x6Q7c9RyMzk"},

{"name":"Arabic Kuthu",
"link":"https://www.youtube.com/watch?v=KUN5Uf9mObQ"},

{"name":"Enjoy Enjaami",
"link":"https://www.youtube.com/watch?v=eYq7WapuDLU"},

{"name":"Kutty Story",
"link":"https://www.youtube.com/watch?v=6ld5y6v5GgA"},

{"name":"Appadi Podu",
"link":"https://www.youtube.com/watch?v=7y2sU6bYb6k"},

{"name":"Jalabulajangu",
"link":"https://www.youtube.com/watch?v=YkW0Y9P6Lx4"},

{"name":"Selfie Pulla",
"link":"https://www.youtube.com/watch?v=6m5XH0Z6k5s"},

{"name":"Chill Bro",
"link":"https://www.youtube.com/watch?v=V2sS6ZsKxZk"},

{"name":"Donu Donu Donu",
"link":"https://www.youtube.com/watch?v=V2sS6ZsKxZk"}

],
        "books": [
            {"name":"The Alchemist","link":"https://www.goodreads.com/book/show/865.The_Alchemist"},
{"name":"Atomic Habits","link":"https://www.goodreads.com/book/show/40121378-atomic-habits"},
{"name":"Think and Grow Rich","link":"https://www.goodreads.com/book/show/30186948-think-and-grow-rich"},
{"name":"Rich Dad Poor Dad","link":"https://www.goodreads.com/book/show/69571.Rich_Dad_Poor_Dad"},
{"name":"Ikigai","link":"https://www.goodreads.com/book/show/40534545-ikigai"},
{"name":"Power of Now","link":"https://www.goodreads.com/book/show/6708.The_Power_of_Now"},
{"name":"Deep Work","link":"https://www.goodreads.com/book/show/25744928-deep-work"},
{"name":"Start With Why","link":"https://www.goodreads.com/book/show/7108725-start-with-why"},
{"name":"Mindset","link":"https://www.goodreads.com/book/show/40745.Mindset"},
{"name":"Grit","link":"https://www.goodreads.com/book/show/27213329-grit"}
        ],
        "games": [
            {"name":"2048","link":"https://play2048.co"},
{"name":"Skribbl","link":"https://skribbl.io"},
{"name":"Sudoku","link":"https://sudoku.com"},
{"name":"Tetris","link":"https://tetris.com/play-tetris"},
{"name":"Chess","link":"https://www.chess.com"},
{"name":"Akinator","link":"https://en.akinator.com"},
{"name":"Pacman","link":"https://pacman.live"},
{"name":"Snake Game","link":"https://playsnake.org"},
{"name":"Crossword","link":"https://nytimes.com/crosswords"},
{"name":"Solitaire","link":"https://solitaire.gg"}
        ],
        "color": "Yellow"
    },

    "Sad": {
        "music":[

{"name":"Kannazhaga",
"link":"https://www.youtube.com/watch?v=E4n7dP3vH2k"},

{"name":"Po Nee Po",
"link":"https://www.youtube.com/watch?v=7Jp4pC5q1u4"},

{"name":"Unna Nenachu",
"link":"https://www.youtube.com/watch?v=7VZP7E1OQ0A"},

{"name":"New York Nagaram",
"link":"https://www.youtube.com/watch?v=Qh4rU5H1E3A"},

{"name":"Munbe Vaa",
"link":"https://www.youtube.com/watch?v=4uJY0F0sYvA"},

{"name":"Maruvaarthai",
"link":"https://www.youtube.com/watch?v=2A4C5qP8F2E"},

{"name":"Nenjukkul Peidhidum",
"link":"https://www.youtube.com/watch?v=8n5dJ0o0qK8"},

{"name":"Thalli Pogathey",
"link":"https://www.youtube.com/watch?v=ZK5p5sBz6cA"},

{"name":"Yennai Maatrum Kadhale",
"link":"https://www.youtube.com/watch?v=bH3N5JkG8tQ"},

{"name":"Oru Naalil",
"link":"https://www.youtube.com/watch?v=2ZqV9pG3F8g"}

],
        "books": [
          {"name":"The Power of Now","link":"https://www.goodreads.com/book/show/6708.The_Power_of_Now"},
{"name":"Feeling Good","link":"https://www.goodreads.com/book/show/46674.Feeling_Good"},
{"name":"You Can Heal Your Life","link":"https://www.goodreads.com/book/show/129603.You_Can_Heal_Your_Life"},
{"name":"Man's Search for Meaning","link":"https://www.goodreads.com/book/show/4069.Man_s_Search_for_Meaning"},
{"name":"The Secret","link":"https://www.goodreads.com/book/show/52529.The_Secret"},
{"name":"Ikigai","link":"https://www.goodreads.com/book/show/40534545-ikigai"},
{"name":"Grit","link":"https://www.goodreads.com/book/show/27213329-grit"},
{"name":"Mindset","link":"https://www.goodreads.com/book/show/40745.Mindset"},
{"name":"Atomic Habits","link":"https://www.goodreads.com/book/show/40121378-atomic-habits"},
{"name":"Deep Work","link":"https://www.goodreads.com/book/show/25744928-deep-work"}
        ],
        "games": [
            {"name":"Sudoku","link":"https://sudoku.com"},
{"name":"2048","link":"https://play2048.co"},
{"name":"Chess","link":"https://chess.com"},
{"name":"Tetris","link":"https://tetris.com"},
{"name":"Snake Game","link":"https://playsnake.org"},
{"name":"Pacman","link":"https://pacman.live"},
{"name":"Solitaire","link":"https://solitaire.gg"},
{"name":"Crossword","link":"https://nytimes.com/crosswords"},
{"name":"Akinator","link":"https://en.akinator.com"},
{"name":"Skribbl","link":"https://skribbl.io"}
        ],
        "color": "Blue"
    },

    "Surprise": {
        "music":[

{"name":"Why This Kolaveri Di",
"link":"https://www.youtube.com/watch?v=YR12Z8f1Dh8"},

{"name":"Google Google",
"link":"https://www.youtube.com/watch?v=Z1BCujX3pw8"},

{"name":"Boomi Enna Suthudhe",
"link":"https://www.youtube.com/watch?v=pYxZbJ4n5UQ"},

{"name":"Oh Penne",
"link":"https://www.youtube.com/watch?v=7t2q9sB2mP0"},

{"name":"Taxi Taxi",
"link":"https://www.youtube.com/watch?v=Jt4sH9gP3kQ"},

{"name":"Vaaya En Veera",
"link":"https://www.youtube.com/watch?v=8q8jQ7tG5mE"},

{"name":"Pakkam Vanthu",
"link":"https://www.youtube.com/watch?v=2z4pY3P7U9M"},

{"name":"Mental Manadhil",
"link":"https://www.youtube.com/watch?v=2Vv-BfVoq4g"},

{"name":"Karuthavanlaam Galeejaam",
"link":"https://www.youtube.com/watch?v=Q9Fh8Z9P3qA"},

{"name":"Adchi Thooku",
"link":"https://www.youtube.com/watch?v=H0z7J9R5F6k"}

],
        "books": [
           {"name":"Atomic Habits","link":"https://www.goodreads.com/book/show/40121378-atomic-habits"},
{"name":"The 5 AM Club","link":"https://www.goodreads.com/book/show/33951933-the-5am-club"},
{"name":"Think and Grow Rich","link":"https://www.goodreads.com/book/show/30186948-think-and-grow-rich"},
{"name":"Start With Why","link":"https://www.goodreads.com/book/show/7108725-start-with-why"},
{"name":"Deep Work","link":"https://www.goodreads.com/book/show/25744928-deep-work"},
{"name":"Mindset","link":"https://www.goodreads.com/book/show/40745.Mindset"},
{"name":"Grit","link":"https://www.goodreads.com/book/show/27213329-grit"},
{"name":"Ikigai","link":"https://www.goodreads.com/book/show/40534545-ikigai"},
{"name":"Rich Dad Poor Dad","link":"https://www.goodreads.com/book/show/69571.Rich_Dad_Poor_Dad"},
{"name":"The Alchemist","link":"https://www.goodreads.com/book/show/865.The_Alchemist"} 
        ],
        "games": [
          {"name":"2048","link":"https://play2048.co"},
{"name":"Chess","link":"https://www.chess.com"},
{"name":"Tetris","link":"https://tetris.com/play-tetris"},
{"name":"Sudoku","link":"https://sudoku.com"},
{"name":"Pacman","link":"https://pacman.live"},
{"name":"Snake Game","link":"https://playsnake.org"},
{"name":"Crossword","link":"https://nytimes.com/crosswords"},
{"name":"Solitaire","link":"https://solitaire.gg"},
{"name":"Akinator","link":"https://en.akinator.com"},
{"name":"Skribbl","link":"https://skribbl.io"} 
        ],
        "color": "Orange"
    },

    "Neutral": {
        "music":[

{"name":"Nenjukkul Peidhidum",
"link":"https://www.youtube.com/watch?v=8n5dJ0o0qK8"},

{"name":"Pachai Nirame",
"link":"https://www.youtube.com/watch?v=0Y4JtS9fF2U"},

{"name":"Ava Enna",
"link":"https://www.youtube.com/watch?v=1Z3E3F5cG7Q"},

{"name":"Vaseegara",
"link":"https://www.youtube.com/watch?v=1P3p8y0V7lE"},

{"name":"Anbil Avan",
"link":"https://www.youtube.com/watch?v=6bM3YJ1P9vE"},

{"name":"Hosanna",
"link":"https://www.youtube.com/watch?v=5p6K1QpF7eU"},

{"name":"Thuli Thuli",
"link":"https://www.youtube.com/watch?v=6B1N3V1dP0E"},

{"name":"Idhazhin Oram",
"link":"https://www.youtube.com/watch?v=4F8Y1mG3k5U"},

{"name":"Moongil Thottam",
"link":"https://www.youtube.com/watch?v=V7g6b6GJm5s"},

{"name":"Munbe Vaa",
"link":"https://www.youtube.com/watch?v=4uJY0F0sYvA"}

],
        "books": [
           {"name":"Atomic Habits","link":"https://www.goodreads.com/book/show/40121378-atomic-habits"},
{"name":"Ikigai","link":"https://www.goodreads.com/book/show/40534545-ikigai"},
{"name":"Deep Work","link":"https://www.goodreads.com/book/show/25744928-deep-work"},
{"name":"The Power of Now","link":"https://www.goodreads.com/book/show/6708.The_Power_of_Now"},
{"name":"Mindset","link":"https://www.goodreads.com/book/show/40745.Mindset"},
{"name":"Grit","link":"https://www.goodreads.com/book/show/27213329-grit"},
{"name":"The Alchemist","link":"https://www.goodreads.com/book/show/865.The_Alchemist"},
{"name":"Rich Dad Poor Dad","link":"https://www.goodreads.com/book/show/69571.Rich_Dad_Poor_Dad"},
{"name":"Start With Why","link":"https://www.goodreads.com/book/show/7108725-start-with-why"},
{"name":"Think and Grow Rich","link":"https://www.goodreads.com/book/show/30186948-think-and-grow-rich"}
        ],
        "games": [
            {"name":"Sudoku","link":"https://sudoku.com"},
{"name":"2048","link":"https://play2048.co"},
{"name":"Chess","link":"https://chess.com"},
{"name":"Tetris","link":"https://tetris.com"},
{"name":"Snake Game","link":"https://playsnake.org"},
{"name":"Pacman","link":"https://pacman.live"},
{"name":"Solitaire","link":"https://solitaire.gg"},
{"name":"Crossword","link":"https://nytimes.com/crosswords"},
{"name":"Akinator","link":"https://en.akinator.com"},
{"name":"Skribbl","link":"https://skribbl.io"}
        ],
        "color": "Green"
    },

    "Fear": {
       "music":[

{"name":"Psycho Saiyaan",
"link":"https://www.youtube.com/watch?v=FjM5Vf0E8nM"},

{"name":"Naan Nee",
"link":"https://www.youtube.com/watch?v=2tXHn9L4Q5g"},

{"name":"Neeye Oli",
"link":"https://www.youtube.com/watch?v=JdR7S9f4c3E"},

{"name":"Uyire",
"link":"https://www.youtube.com/watch?v=Vf4pJ1f8E6A"},

{"name":"Kadhal Rojave",
"link":"https://www.youtube.com/watch?v=5pHk9Kj2JgA"},

{"name":"Mazhai Kuruvi",
"link":"https://www.youtube.com/watch?v=Z6b4R8X1F7Y"},

{"name":"Aaromale",
"link":"https://www.youtube.com/watch?v=K4DyBUG242c"},

{"name":"Suttrum Vizhi",
"link":"https://www.youtube.com/watch?v=5GvR7mJc5sQ"},

{"name":"Vinnaithaandi Varuvaya Theme",
"link":"https://www.youtube.com/watch?v=ZP9YJ7A5m5A"},

{"name":"Omana Penne",
"link":"https://www.youtube.com/watch?v=9M2pH1F0cE0"}

],
        "books": [
            {"name":"Feel the Fear and Do It Anyway","link":"https://www.goodreads.com/book/show/653396.Feel_the_Fear_and_Do_It_Anyway"},
{"name":"The Power of Now","link":"https://www.goodreads.com/book/show/6708.The_Power_of_Now"},
{"name":"Atomic Habits","link":"https://www.goodreads.com/book/show/40121378-atomic-habits"},
{"name":"Mindset","link":"https://www.goodreads.com/book/show/40745.Mindset"},
{"name":"Grit","link":"https://www.goodreads.com/book/show/27213329-grit"},
{"name":"Ikigai","link":"https://www.goodreads.com/book/show/40534545-ikigai"},
{"name":"Man's Search for Meaning","link":"https://www.goodreads.com/book/show/4069.Man_s_Search_for_Meaning"},
{"name":"Deep Work","link":"https://www.goodreads.com/book/show/25744928-deep-work"},
{"name":"Start With Why","link":"https://www.goodreads.com/book/show/7108725-start-with-why"},
{"name":"Think and Grow Rich","link":"https://www.goodreads.com/book/show/30186948-think-and-grow-rich"}
        ],
        "games": [
           {"name":"Sudoku","link":"https://sudoku.com"},
{"name":"2048","link":"https://play2048.co"},
{"name":"Chess","link":"https://chess.com"},
{"name":"Tetris","link":"https://tetris.com"},
{"name":"Snake Game","link":"https://playsnake.org"},
{"name":"Pacman","link":"https://pacman.live"},
{"name":"Solitaire","link":"https://solitaire.gg"},
{"name":"Crossword","link":"https://nytimes.com/crosswords"},
{"name":"Akinator","link":"https://en.akinator.com"},
{"name":"Skribbl","link":"https://skribbl.io"}
        ],
        "color": "Purple"
    }
}

# -------------------------------
# SAVE EMOTION
# -------------------------------
def save_emotion(emotion):

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO emotions (emotion, date) VALUES (?, datetime('now'))",
        (emotion,)
    )

    conn.commit()
    conn.close()

# -------------------------------
# HOME PAGE
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html")

# -------------------------------
# EMOTION DETECTION
# -------------------------------
@app.route('/detect')
def detect():

    cap = cv2.VideoCapture(0)
    detected_emotion = "Neutral"

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
            face = face/255.0
            face = np.reshape(face,(1,48,48,1))

            prediction = model.predict(face)
            emotion_index = np.argmax(prediction)
            detected_emotion = emotion_labels[emotion_index]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(frame,detected_emotion,(x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        cv2.imshow("Emotion Detection",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_emotion(detected_emotion)

    rec = recommendations.get(detected_emotion,recommendations["Neutral"])

    return render_template("result.html",
                           emotion=detected_emotion,
                           data=rec)

# -------------------------------
# DATABASE CHECK
# -------------------------------
@app.route('/check')
def check():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM emotions")
    data = cursor.fetchall()

    conn.close()

    return str(data)

# -------------------------------
# LIVE GRAPH DATA
# -------------------------------
@app.route('/emotion-data')
def emotion_data():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    SELECT emotion, COUNT(*)
    FROM emotions
    GROUP BY emotion
    """)

    rows = cursor.fetchall()

    conn.close()

    result = {}

    for row in rows:
        result[row[0].lower()] = row[1]

    return jsonify(result)

# -------------------------------
# DASHBOARD
# -------------------------------
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

# -------------------------------
# HISTORY PAGE
# -------------------------------
@app.route("/music/<emotion>")
def music_page(emotion):

    rec = recommendations.get(emotion)

    return render_template(
        "details.html",
        title=f"{emotion} Music Recommendations",
        items=rec["music"]
    )


@app.route("/books/<emotion>")
def books_page(emotion):

    rec = recommendations.get(emotion)

    return render_template(
        "details.html",
        title=f"{emotion} Book Recommendations",
        items=rec["books"]
    )


@app.route("/games/<emotion>")
def games_page(emotion):

    rec = recommendations.get(emotion)

    return render_template(
        "details.html",
        title=f"{emotion} Game Recommendations",
        items=rec["games"]
    )
@app.route('/history')
def history():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM emotions ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    return render_template("history.html",data=data)

# -------------------------------
# FEEDBACK
# -------------------------------
@app.route('/feedback',methods=["POST"])
def feedback():

    user_feedback = request.form["feedback"]

    return render_template("thankyou.html",message=user_feedback)

# -------------------------------
# PDF REPORT DOWNLOAD
# -------------------------------
@app.route('/download/<emotion>')
def download_report(emotion):

    rec = recommendations.get(emotion,recommendations["Neutral"])

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial",size=16)
    pdf.cell(200,10,"AI Emotion Detection Report",ln=True)

    pdf.ln(10)

    pdf.set_font("Arial",size=12)

    pdf.cell(200,10,f"Detected Emotion: {emotion}",ln=True)
    pdf.cell(200,10,f"Color Therapy: {rec['color']}",ln=True)

    pdf.ln(10)

    pdf.cell(200,10,"Music Recommendation:",ln=True)
    for m in rec["music"]:
        pdf.cell(200,10,m["name"],ln=True)

    pdf.ln(5)

    pdf.cell(200,10,"Book Recommendation:",ln=True)
    for b in rec["books"]:
        pdf.cell(200,10,b["name"],ln=True)

    pdf.ln(5)

    pdf.cell(200,10,"Game Recommendation:",ln=True)
    for g in rec["games"]:
        pdf.cell(200,10,g["name"],ln=True)

    file_name = "emotion_report.pdf"

    pdf.output(file_name)

    return send_file(file_name, as_attachment=True)

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
