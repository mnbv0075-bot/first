import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 데이터 읽기 (네이버 영화 리뷰 15만 개)
print("데이터 읽는 중...")
df = pd.read_csv("data/ratings_train.txt", sep="\t")
df = df.dropna()  # 내용이 비어있는 리뷰는 버리기

X = df["document"]  # 리뷰 내용
y = df["label"]     # 정답 (0=부정, 1=긍정)

# 2. 연습문제 80% / 시험문제 20%로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 글자를 숫자로 바꾸는 도구 만들기
# 한국어는 "재밌다/재밌어요"처럼 끝이 변하니까
# 글자 조각(1~3글자) 단위로 쪼개서 분석해요
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(1, 3),
    max_features=50000
)
print("텍스트를 숫자로 변환 중...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. 모델 학습 (여기서 시간이 좀 걸려요)
print("학습 중... 몇 분 걸릴 수 있어요!")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. 시험 보기
accuracy = model.score(X_test_vec, y_test)
print(f"정확도: {accuracy:.4f}")

# 6. 저장하기 (앱이 이 파일들을 읽어서 사용해요)
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("저장 완료! 이제 앱을 실행해보세요.")