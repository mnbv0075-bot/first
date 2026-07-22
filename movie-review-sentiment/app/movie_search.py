import joblib
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from deep_translator import GoogleTranslator
app = FastAPI()

model = joblib.load("../model/sentiment_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

TMDB_API_KEY =  "e7555575912db616b2f3dc5f671a2b65"   # ← main_tmdb.py 14번째 줄에 있는 키 복사!


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>영화 리뷰 종합 분석기</title>
<style>
  body { font-family: sans-serif; background: #1a1a2e; color: white;
         max-width: 700px; margin: 40px auto; padding: 20px; }
  h1 { text-align: center; }
  input { width: 75%; padding: 14px; font-size: 16px; border-radius: 10px;
          border: none; box-sizing: border-box; }
  button { width: 23%; padding: 14px; font-size: 16px; background: #e94560;
           color: white; border: none; border-radius: 10px; cursor: pointer; }
  #result { margin-top: 20px; padding: 20px; border-radius: 10px;
            background: #16213e; display: none; }
  .verdict { font-size: 26px; text-align: center; margin: 15px 0; }
  .bar { height: 25px; border-radius: 12px; background: #c0392b;
         overflow: hidden; margin: 10px 0; }
  .bar-fill { height: 100%; background: #27ae60; }
  .review-item { padding: 10px; margin: 6px 0; border-radius: 6px;
                 background: #0f3460; font-size: 13px; }
  img { border-radius: 10px; display: block; margin: 0 auto; }
</style>
</head>
<body>
  <h1>🎬 영화 리뷰 종합 분석기</h1>
  <p>영화 제목을 검색하면 리뷰를 모아서 자동으로 분석해요! (한글 제목 가능)</p>

  <input id="title" placeholder="예: 인셉션, 기생충, Avengers"
         onkeydown="if(event.key==='Enter') search()">
  <button onclick="search()">검색 🔍</button>

  <div id="result"></div>

<script>
async function search() {
  const title = document.getElementById("title").value;
  if (!title.trim()) { alert("영화 제목을 입력해주세요!"); return; }

  const box = document.getElementById("result");
  box.style.display = "block";
  box.innerHTML = "⏳ 리뷰 수집하고 분석하는 중...";

  const res = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title })
  });
  const data = await res.json();

  if (data.error) { box.innerHTML = "❌ " + data.error; return; }

  let itemsHtml = "";
  data.details.forEach(d => {
    itemsHtml += `<div class="review-item">${d.emoji} ${d.review}...</div>`;
  });

  box.innerHTML = `
    ${data.poster ? `<img src="${data.poster}" width="180">` : ""}
    <h2 style="text-align:center">${data.movie_title} (${data.year})</h2>
    <div class="verdict">${data.verdict}</div>
    <div>📊 리뷰 ${data.total}개 분석 완료</div>
    <div class="bar"><div class="bar-fill" style="width:${data.ratio}%"></div></div>
    <div>😍 긍정 ${data.positive}개 (${data.ratio}%) | 😡 부정 ${data.negative}개</div>
    <hr>
    ${itemsHtml}
  `;
}
</script>
</body>
</html>
"""


class SearchInput(BaseModel):
    title: str


@app.post("/search")
def search(data: SearchInput):
    # 1. 영화 제목으로 검색 (한글 제목도 인식!)
    search_res = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": TMDB_API_KEY, "query": data.title, "language": "ko-KR"}
    ).json()

    if not search_res.get("results"):
        return {"error": "영화를 찾을 수 없어요. 제목을 다시 확인해주세요!"}

    movie = search_res["results"][0]  # 가장 관련 높은 영화
    movie_id = movie["id"]

    # 2. 그 영화의 리뷰 가져오기 (1~3페이지)
    reviews = []
    for page in [1, 2, 3]:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/reviews",
            params={"api_key": TMDB_API_KEY, "page": page}
        ).json()
        for item in r.get("results", []):
            reviews.append(item["content"])
        if page >= r.get("total_pages", 1):
            break

    if len(reviews) == 0:
        return {"error": f"'{movie['title']}' 영화는 등록된 리뷰가 없어요. 더 유명한 영화로 해보세요!"}

    # 3. 전부 분석!
    X = vectorizer.transform(reviews)
    predictions = model.predict(X)

    total = len(predictions)
    positive = int(predictions.sum())
    negative = total - positive
    ratio = round(positive / total * 100, 1)

    if total < 5:
        verdict = "🤷 리뷰가 부족해서 판단 보류!"
    elif ratio >= 80:
        verdict = "🔥 강력 호평작!"
    elif ratio >= 60:
        verdict = "👍 대체로 호평"
    elif ratio >= 40:
        verdict = "🤔 호불호 갈리는 영화"
    else:
        verdict = "💀 혹평작..."

    translator = GoogleTranslator(source="en", target="ko")
    details = []
    for r, p in zip(reviews[:10], predictions[:10]):  # 번역은 10개까지만 (속도 때문)
        try:
            korean = translator.translate(r[:150])
        except Exception:
            korean = r[:150]  # 번역 실패하면 그냥 영어로
        details.append({"review": korean, "emoji": "😍" if p == 1 else "😡"})

    return {
        "movie_title": movie["title"],
        "year": (movie.get("release_date") or "????")[:4],
        "poster": f"https://image.tmdb.org/t/p/w300{movie['poster_path']}" if movie.get("poster_path") else None,
        "total": total, "positive": positive, "negative": negative,
        "ratio": ratio, "verdict": verdict, "details": details
    }