import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import subprocess

# aclImdb 폴더를 자동으로 찾기
DATA_DIR = subprocess.run(
    ["find", "/workspaces/first", "-type", "d", "-name", "aclImdb"],
    capture_output=True, text=True
).stdout.strip().split("\n")[0]

print(f"데이터 폴더 찾음: {DATA_DIR}")

def load_reviews(folder):
    texts, labels = [], []
    for label_name, label in [("pos", 1), ("neg", 0)]:
        path = os.path.join(folder, label_name)
        for fname in os.listdir(path):
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(label)
    return texts, labels

print("데이터 읽는 중... (1~2분 걸려요)")
train_texts, train_labels = load_reviews(os.path.join(DATA_DIR, "train"))
test_texts, test_labels = load_reviews(os.path.join(DATA_DIR, "test"))
print(f"학습 데이터: {len(train_texts)}개, 테스트 데이터: {len(test_texts)}개")

print("텍스트 벡터화 중...")
vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print("모델 학습 중...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

acc = accuracy_score(test_labels, model.predict(X_test))
print(f"🎯 정확도: {acc * 100:.1f}%")

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("모델 저장 완료! (sentiment_model.pkl, vectorizer.pkl)")