import joblib
import urllib.request
import urllib.parse
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("../model/sentiment_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

TMDB_API_KEY = "e7555575912db616b2f3dc5f671a2b65"  # ← 어제 발급받은 짧은 v3 키!

class Review(BaseModel):
    text: str
    title: str = ""

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>영화 리뷰 감성 분석</title></head>
    <body style="font-family:sans-serif; max-width:600px; margin:50px auto;">
        <h1>🎬 영화 리뷰 감성 분석</h1>
        <input id="title" style="width:100%; font-size:16px; padding:8px;"
            placeholder="영화 제목 (한글 가능, 예: 기생충)"><br><br>
        <textarea id="review" rows="5" style="width:100%; font-size:16px;"
            placeholder="리뷰 입력 (예: 진짜 재밌었어요!)"></textarea><br><br>
        <button onclick="analyze()" style="font-size:18px; padding:10px;">분석하기</button>
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
                    "<img src='" + data.poster + "' style='max-width:300px;'>";
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
    if prob < 65:
        result = f"🤔 애매한 리뷰입니다 (확신도 {prob:.0f}%)"
    elif pred == 1:
        result = f"😊 긍정 리뷰입니다! (확신도 {prob:.0f}%)"
    else:
        result = f"😞 부정 리뷰입니다! (확신도 {prob:.0f}%)" 

    response = {"result": result, "poster": None}

    # 영화 제목이 입력됐으면 TMDB에서 검색 (한국어 지원!)
    if review.title.strip():
        try:
            query = urllib.parse.quote(review.title)
            url = (f"https://api.themoviedb.org/3/search/movie"
                   f"?api_key={TMDB_API_KEY}&query={query}&language=ko-KR")
            with urllib.request.urlopen(url) as r:
                data = json.loads(r.read())
            if data["results"]:
                movie = data["results"][0]  # 검색 결과 첫 번째 영화
                response["movie_title"] = movie["title"]
                response["year"] = movie.get("release_date", "")[:4]
                if movie.get("poster_path"):
                    response["poster"] = "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
        except Exception:
            pass  # 포스터 검색 실패해도 감성 분석 결과는 보여줌

    return response