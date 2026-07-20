import urllib.request
import tarfile

url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
print('다운로드 시작...')
urllib.request.urlretrieve(url, 'aclImdb_v1.tar.gz')
print('다운로드 완료!')

print('압축 풀는 중...')
with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()
print('압축 풀기 완료!')
print('aclImdb 폴더 생성 완료')