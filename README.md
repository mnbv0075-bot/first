# 영화 리뷰 감성분류 프로젝트

IMDB 영화 리뷰 데이터를 활용하여 
긍정/부정을 판단하는 AI 모델을 만드는 프로젝트입니다.

## 폴더 구조

- `data/` - 영화 리뷰 데이터
- `model/` - AI 학습 코드
- `app/` - 웹 서비스 코드

## 사용 기술

- Python
- scikit-learn 또는 LSTM
- FastAPI


# 🎬 영화 리뷰 감성 분석 & 종합 판정 서비스

영화 제목을 검색하면 **실제 리뷰를 자동으로 수집**하고, 직접 학습시킨 AI 모델로
**긍정/부정을 분석**해서 영화의 종합 평가를 알려주는 웹 서비스입니다.

## 📌 주요 기능

- **리뷰 감성 분석**: 리뷰 텍스트를 입력하면 긍정/부정 자동 판별
- **영화 검색 종합 판정**: 영화 제목 검색 → TMDB API로 실제 리뷰 수집 → 전체 리뷰 일괄 분석
- **종합 판정 시스템**: 긍정 비율에 따라 등급 판정
  - 80% 이상 → 🍿 강력 호평작!
  - 60~80% → 👍 대체로 호평
  - 40~60% → 🤔 호불호 갈리는 영화
  - 40% 미만 → 💀 혹평작...
- **한국어 번역**: 영어 리뷰를 한국어로 번역해서 표시 (deep-translator)
- **포스터/개봉연도 표시**: 검색한 영화의 정보를 함께 제공

## 🤖 AI 모델

| 항목 | 내용 |
|------|------|
| 학습 데이터 | IMDB 영화 리뷰 50,000개 (aclImdb) |
| 벡터화 | TF-IDF (max_features=20,000) |
| 모델 | Logistic Regression |
| **정확도** | **88.0%** |

## 🛠 사용 기술

- **Python** - 전체 개발 언어
- **scikit-learn** - 감성 분류 모델 학습 (TF-IDF + LogisticRegression)
- **FastAPI** - 웹 서버 및 API 구현
- **TMDB API** - 영화 정보 및 리뷰 수집
- **deep-translator** - 영어 리뷰 한국어 번역


## 🚀 실행 방법

```bash
# 1. 필요한 라이브러리 설치
pip install -r requirements.txt

# 2. 모델 학습 (model 폴더에서)
cd model
python train.py

# 3. 서버 실행 (app 폴더에서)
cd ../app
python -m uvicorn movie_search:app --reload

# 4. 브라우저에서 접속
# http://localhost:8000




