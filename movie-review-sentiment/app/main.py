import os
import joblib
import urllib.request
import urllib.parse
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "..", "model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl"))

OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "") # ← 여기에 이메일로 받은 키 입력!

class Review(BaseModel):
    text: str
    title: str = ""

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>영화 리뷰 감성 분석</title></head>
    <body style="font-family:sans-serif; max-width:600px; margin:50px auto; text-align:center;">
        <h1>🎬 영화 리뷰 감성 분석</h1>
        <input id="title" style="width:100%; font-size:16px; padding:8px;"
            placeholder="영화 제목 (영어로, 예: Avatar)"><br><br>
        <textarea id="review" rows="5" style="width:100%; font-size:16px;"
            placeholder="영어로 리뷰 입력 (예: This movie was amazing!)"></textarea><br><br>
        <button onclick="analyze()" style="font-size:18px; padding:10px 30px;">분석하기</button>
        <h2 id="result"></h2>
        <div id="movie"></div>
        <script>
        async function analyze() {
            const text = document.getElementById("review").value;
            if (!text.trim()) { alert("리뷰를 입력해주세요!"); return; }
            const title = document.getElementById("title").value;
            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text: text, title: title})
            });
            const data = await res.json();
            document.getElementById("result").innerText = data.result;
            if (data.poster) {
                document.getElementById("movie").innerHTML =
                    "<h3>" + data.movie_title + " (" + data.year + ")</h3>" +
                    "<img src='" + data.poster + "' style='max-width:300px; border-radius:10px;'>";
            } else {
                document.getElementById("movie").innerHTML = "";
            }
        }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict(review: Review):
    X = vectorizer.transform([review.text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred] * 100
    if pred == 1:
        result = f"😊 긍정 리뷰입니다! (확신도 {prob:.0f}%)"
    else:
        result = f"😞 부정 리뷰입니다! (확신도 {prob:.0f}%)"

    response = {"result": result, "poster": None}

    # 영화 제목이 입력됐으면 포스터 검색
    if review.title.strip():
        try:
            url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={urllib.parse.quote(review.title)}"
            with urllib.request.urlopen(url) as r:
                movie = json.loads(r.read())
            if movie.get("Response") == "True" and movie.get("Poster") != "N/A":
                response["poster"] = movie["Poster"]
                response["movie_title"] = movie["Title"]
                response["year"] = movie["Year"]
        except Exception:
            pass  # 포스터 검색 실패해도 감성 분석 결과는 보여줌

    return response