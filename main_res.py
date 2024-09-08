#путь к модели tf-idf
path_tdidf = 'tfidf.pkl'
#путь к модели предсказания профессии
path_logit = 'logit.pkl'

#путь к данным резюме
path_data ='vprod_test/TEST_RES.csv'
#путь к создаваемому сабмишену
path_submit = 'submission_RES_part.csv'


import pandas as pd
import numpy as np
import joblib


import re
from tqdm.notebook import tqdm
tqdm.pandas()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')
import spacy

stopwords_nltk = nltk.corpus.stopwords.words('russian')
stopwords_nltk_en = nltk.corpus.stopwords.words('english')
stopwords_nltk.extend(stopwords_nltk_en)

new_stop = ["обязанность", "должностной", "работать", "инструкция", "работа", "согласно", "должность"]
stopwords_nltk.extend(new_stop)

lemmatizer = spacy.load('ru_core_news_md', disable = ['parser', 'ner'])


# очистка текста
def full_clean(text):
    '''подготовка текста к подаче в модель для каждого текста'''
    try:
        text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9#]", " ", text)
        text = text.lower()
        text = re.sub(" +", " ", text).strip()  # оставляем только 1 пробел
    except:
        text = ' '
    # токены для моделей
    tokens = [token.lemma_ for token in lemmatizer(text) if token.lemma_ not in stopwords_nltk]
    # для tfidf на вход текст
    text = " ".join(tokens)
    return text


def preprocess_text(df):
    '''подготовка текста к подаче в модель колонкой'''
    new_corpus = []

    for text in tqdm(df):
        text = full_clean(text)
        new_corpus.append(text)

    return new_corpus


def tfidf_embeding(model=None, df=None):
    '''Преобразование текста в мешок слов'''
    if model == None:
        # загрузить если нет
        model = joblib.load(path_tdidf)
    else:
        model = model
    X = model.transform(df)
    return X.toarray()

def create_submission_part(test_df, name_of_predict_column, value):
    '''создание сабмишена под задачу'''
    submission = pd.DataFrame([])
    submission['id'] = test_df['id']
    submission[name_of_predict_column] = value
    return submission

logit = LogisticRegression()
tfidf = TfidfVectorizer()

tfidf = joblib.load(path_tdidf)
logit = joblib.load(path_logit)

df_test_RES = pd.read_csv(path_data)

df_test_RES['text_clean'] = preprocess_text(df_test_RES['demands'])

tfidf_test = tfidf_embeding(model=tfidf, df=df_test_RES['text_clean'])

model_test = logit.predict(tfidf_test)

submission_RES_part = create_submission_part(df_test_RES, 'job_title', model_test)

submission_RES_part.to_csv(path_submit, index = False)

