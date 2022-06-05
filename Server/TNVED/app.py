import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
from PIL import Image
from nltk.stem import SnowballStemmer
import json
import re
from io import StringIO
import collections

X_TfidfVectorizer = pd.DataFrame()
IMAGE_HEADER = Image.open('header.png')
IMAGE_QR = Image.open('QR_code.png')
col1, col2, col3, col4 = st.columns(4)
col1.image(IMAGE_HEADER)
col2.metric(label="Точность тoп 1", value="68.7%", delta="6.7%")
col3.metric(label="Точность топ 5", value="88.2%", delta="7.1%")
col4.image(IMAGE_QR)

stemmer_ru = SnowballStemmer("russian")


def clean(text):
    text = text.lower()
    text = text + ' '
    text = text.replace('не ', 'не')
    text = text.replace('без ', 'без')
    text = text.replace(' назначения', 'назначения')
    text = text.replace(' имеющие ', ' имеющие')

    text = re.sub('\W', ' ', text)  # Любая не-буква, не-цифра и не подчёркивание
    text = re.sub('\d', ' ', text)  # Любая цифра

    text = text.replace(' и ', ' ')
    text = text.replace(' но ', ' ')
    text = text.replace(' для ', ' ')
    text = text.replace(' из ', ' ')
    text = text.replace(' поз ', ' ')
    text = text.replace(' как ', ' ')

    text = text.replace(' более ', ' ')
    text = text.replace(' менее ', ' ')
    text = text.replace(' неболее ', ' ')
    text = text.replace(' неменее ', ' ')
    text = text.replace(' всего ', ' ')

    text = text.replace(' ква ', ' ')

    text = re.sub('\W\w{1}\W', ' ', text)  # Одна буква
    text = re.sub('\s\w\s', ' ', text)  # Одна буква
    text = re.sub('\W\w{2}\W', ' ', text)  # две буквы
    text = re.sub('\s\w\w\s', ' ', text)  # две буквы

    text = re.sub(r'[^а-яА-Я\s]+', '', text).lower()
    text = text.split()

    temp = [stemmer_ru.stem(word) for word in text]
    text = ' '.join(temp)
    # print(text,'\n')
    return text


@st.cache(allow_output_mutation=True)
def load_voc(vocabulary_file):
    with open(vocabulary_file, 'r') as f:
        voc = json.load(f)
    return voc


@st.cache(allow_output_mutation=True)
def load_model(model_file):
    with open(model_file, 'rb') as f:
        knn = pickle.load(f)
    return knn


@st.cache(allow_output_mutation=True)
def create_vectorizer(voc):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, vocabulary=voc)
    return vectorizer


@st.cache(allow_output_mutation=True)
def load_indexes(indexes_file):
    with open(indexes_file, 'rb') as f:
        indexes = json.load(f)
    return indexes


def vectorize(vectorizer, data, voc):
    X_TfidfVectorizer = vectorizer.fit_transform([data])
    X_TfidfVectorizer = pd.DataFrame(X_TfidfVectorizer.toarray(), columns=voc)
    return X_TfidfVectorizer


def get_value(indexes, predicts):
    array = []
    for i in predicts[0]:
        array.append(indexes[int(i)])
    return array


def test(text):
    st.write(text)


@st.cache(allow_output_mutation=True)
def load_opisanie():
    with open('TNVED_OPISANIE_full.json', 'r') as f:
        opisanie = json.load(f)
    return opisanie


voc = load_voc('voc2.json')
vectorizer = create_vectorizer(voc)
indexes = load_indexes('indexes2.json')
knn = load_model('hacks_ai_top.pkl')
opisanie = load_opisanie()


def main():
    page = st.sidebar.selectbox('Выберите действие: ', ['Подбор \"ТН ВЭД ЕАС\" кода', 'Проверка \"ТН ВЭД ЕАС\" кода'])
    if page == 'Подбор \"ТН ВЭД ЕАС\" кода':
        description = st.text_area('Введите описание: ')
        if st.button('Вывести \"ТН ВЭД ЕАС\"'):
            description = clean(description)
            # test("Нормализация: " + description)
            vector = vectorize(vectorizer, description, voc)
            kneighbors = knn.kneighbors(vector, return_distance=False)
            test_predict = knn.predict(vector)
            predicts = get_value(indexes, kneighbors)
            with st.sidebar:
                st.header('Возможные варианты: ')
                checks = {}
                for i in predicts:
                    checks[i] = i
                count = 0
                for i in checks:
                    check_opisanie = True
                    next_variant = f'{i}'
                    try:
                        next_opisanie = opisanie[next_variant]
                    except:
                        check_opisanie = False
                        next_opisanie = "Описание отсутвует"
                    while len(next_variant) < 4:
                        next_variant = '0' + next_variant

                    if check_opisanie:
                        text_split = next_opisanie.split(' ')
                        count_split = 0
                        if isinstance(text_split, collections.Iterable):
                            for i in text_split:
                                # st.write(i)
                                next_variant = next_variant + " " + i
                                count_split += 1
                                if count_split > 3:
                                    break
                        else:
                            next_variant = next_variant + " " + next_opisanie
                    with st.expander(next_variant):
                        if check_opisanie:
                            st.write(next_opisanie)
                        else:
                            st.error(next_opisanie)
                    count += 1
                    if count > 4:
                        break
                    # st.write(i)

            next_variant = str(test_predict[0])
            try:
                next_opisanie = opisanie[next_variant]
            except:
                next_opisanie = "Описание отсутвует"
            while len(next_variant) < 4:
                next_variant = '0' + next_variant
            st.info(f'Наиболее подходящий вариант: {next_variant}')
            st.write(f'Описание: {next_opisanie}')

    if page == 'Проверка \"ТН ВЭД ЕАС\" кода':
        st.header('Загрузите файл в формате **json**')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            data = json.loads(string_data)
            k = 0
            for next_json in data:
                description = data[next_json]
                next_element = {}
                next_element['описание'] = description
                description = clean(description)

                vector = vectorize(vectorizer, description, voc)
                test_predict = knn.predict(vector)
                next_element['код_для_проверки'] = next_json
                next_element['код'] = str(test_predict[0])
                k += 1
                if next_element['код_для_проверки'] == next_element['код']:
                    st.success(
                        f"Совпадение: \nОписание- {next_element['описание']}\nКод для проверки: {next_element['код_для_проверки']}\nПредсказанный код: {next_element['код']}")
                else:
                    st.error(
                        f"Не Совпадение: \nОписание- {next_element['описание']}\nКод для проверки: {next_element['код_для_проверки']}\nПредсказанный код: {next_element['код']}")

                if k > 10:
                    break


if __name__ == "__main__":
    main()
