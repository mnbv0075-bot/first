import urllib.request
import os

os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
print("한국어 리뷰 데이터 다운로드 중... (약 14MB)")
urllib.request.urlretrieve(url, "data/ratings_train.txt")
print("완료!")