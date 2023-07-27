# 기본 요소들 메모
# pickle, pandas(as pd), numpy(as np), matplotlib.pyplot(as plt) : 파이썬 그래프용
# re, urllib.request, Okt(from konlpy.tag), tqdm(from tqdm)
# Tokenizer(from tensorflow.keras.preprocessing.text)
# pad_sequences(from tensorflow.keras.preprocessing.sequence)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import re
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from tqdm import tqdm
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


from matplotlib import font_manager, rc

font_path = './HYHWPEQ.TTF'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# # 1. github repository에 등록된 txt 파일 안의 텍스트 데이터들을 urllib을 사용해 불러온다.
# urllib.request.urlretrieve("https://raw.githubusercontent.com/woodawon/Recognition_model/main/Texts/ai_train.txt",
#                            filename="ai_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/woodawon/Recognition_model/main/Texts/ai_test.txt",
#                            filename="ai_test.txt")

def remove_special_characters(text):
    # 정규표현식을 사용하여 특수문자를 제거합니다.
    text = re.sub(r'[「」“〔〕ℓ△”]', '', text)
    return text

# 텍스트 파일 경로 지정
file_path = "./ai_train.txt"

# 파일 열기
with open(file_path, "r", encoding="utf-8") as file:
    # 파일 내용 읽어오기
    file_content = file.read()

train_text = remove_special_characters(file_content)
# 파일을 쓰기 모드로 열어서 텍스트를 파일에 씁니다.
with open('ai_train.txt', 'w', encoding='utf-8') as file:
    file.write(train_text)

# 텍스트 파일 경로 지정
file_path = "./ai_test.txt"

# 파일 열기
with open(file_path, "r", encoding="utf-8") as file:
    # 파일 내용 읽어오기
    file_content = file.read()

test_text = remove_special_characters(file_content)
# 파일을 쓰기 모드로 열어서 텍스트를 파일에 씁니다.
with open('ai_test.txt', 'w', encoding='utf-8') as file:
    file.write(test_text)


# 데이터 변수 선언(= 데이터를 변수에 저장)
test_data= pd.read_table('ai_test.txt', delimiter=' ')
train_data = pd.read_table('ai_train.txt', delimiter=' ')

# print(train_data[:5])
# print(len(train_data))
# print(len(test_data))
# print(test_data[:5])


# 2. 데이터 정제

# 2-1. 중복 제거
# 중복이 제거된 후의 document와 label 값의 개수 출력
print(train_data['document'].nunique(), train_data['label'].nunique())
# 중복 제거하기
# subset : 중복값을 제거할 열 / inplace : 원본 변경 여부
train_data.drop_duplicates(subset=['document'], inplace=True)
print("중복 제거 후 document 개수 : ", len(train_data))
# 그래프 그리기
plt.figure(figsize=(10, 5))
train_data['label'].value_counts().plot(kind='bar')
plt.xlabel('label 0 or 1')
plt.ylabel('분포')
plt.title('label 분포')
plt.show()

# 2-2. Null인 샘플 제거
# Null값 제거하기
print(train_data.isnull().values.any())  # True or False : null값 인게 있는지 없는지?
print(train_data.isnull().sum())  # null값이 존재하는 열의 id, document, label의 개수 출력
print(train_data.loc[train_data.document.isnull()])  # null인 샘플이 어느 index 위치에 존재하는지 출력
train_data = train_data.dropna(how='any')  # null인 행 제거
print(train_data.isnull().values.any())  # null인 거 있는지 없는지? True or False 출력

# 2-3. 데이터 전처리
# 정규 표현식으로 특수 문자 등을 제거하기
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
# ++ 디테일 : white space로 인한 emply 값을 Null처리하여 제거해주기
train_data['document'] = train_data['document'].str.replace('^ +', "", regex=True)
# 공백을 Nan(= Null)로 replace 시켜주고, 이 작업으로 데이터를 변경하는 것을 허용시킨다(= inplace = True)
train_data['document'].replace('', np.NaN, inplace=True)
# print(train_data.isnull().sum()) # sum은 더하기가 아니라, 열을 의미함.
train_data = train_data.dropna(how='any')  # 2-2에서 한 것처럼 한 번 더 Null 값들 제거 시키기

# 2-4. 테스트 데이터
test_data.drop_duplicates(subset=['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True)
test_data['document'].replace('', np.NaN, inplace=True)
test_data = test_data.dropna(how='any')

# 3. 토큰화
# 한글 -> 형태소 분석기
okt = Okt()
# 불용어 선언
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
X_train = []
for sentence in tqdm(train_data['document']):  # tqdm이란? : 파이썬 반복문의 진행률 표시함.
    # 토큰화(feat. Okt) / stem=True : 정규화 시키기
    tokenizered_sentence = okt.morphs(sentence, stem=True)
    # 불용어 삭제(feat. stopwords)
    stopwords_removed_sentence = [word for word in tokenizered_sentence if not word in stopwords]
    X_train.append(stopwords_removed_sentence)
print(X_train[:3])
# 3-1. 테스트 데이터 토큰화
X_test = []
for sentence in tqdm(test_data['document']):
    tokenizered_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenizered_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

# 4. 정수 인코딩
# 단어 집합(vocaburary) 만들기
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
hold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < hold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

text_size = total_cnt - rare_cnt + 1
tokenizer = Tokenizer(text_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
print(X_train[:3])
X_test = tokenizer.texts_to_sequences(X_test)
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])


# 5. 빈 샘플 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
print(X_train[:3])
y_train = np.delete(y_train, drop_train, axis=0)


# 6. 패딩
print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_hold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count += 1
    print('%s 이하인 샘플 비율 : %s'%(max_len, (count / len(nested_list))*100))
below_hold_len(30, X_train)
X_train = pad_sequences(X_train, maxlen=30)
X_test = pad_sequences(X_test, maxlen=30)



# 세션에서 모델을 실행하고, 학습 또는 추론 등을 수행합니다.
# LSTM모델로 네이버 영화 리부 감성 분류하기

embedding_dim = 100  # 임베딩 벡터의 차원
hidden_units = 128  # 은닉 상태의 크기
model = Sequential()
model.add(Embedding(text_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))  # 활성화 함수 : sigmoid 함수를 사용하겠다고 선언.

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)
load_models = load_model('best_model.h5')
print("\n 테스트 정확도: %2f" % (load_models.evaluate(X_test, y_test)[1]))


# 리뷰 예측해보기
def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
    padding_new = pad_sequences(encoded, maxlen=30)  # 패딩
    score = float(load_models.predict(padding_new))  # 예측!!
    if (score > 0.5):
        print("{:.2f}% 확률로 환경입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 인권입니다.\n".format((1 - score) * 100))


sentiment_predict('인권은 반드시 보장되어야 할 기본권이야')
sentiment_predict('쓰레기는 치워야 해')
