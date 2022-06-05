import telebot
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
from nltk.stem import SnowballStemmer
import json
import re


bot = telebot.TeleBot('5463576262:AAEoMmqZXCFpZb2vsm3nhSH463Xm6upjWAs')
stemmer_ru = SnowballStemmer("russian")

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Добрый день вас приветствует бот команды INNINO!")
        bot.send_message(message.from_user.id, "Введите ТН ВЭД код чтобы получить его описание")
        bot.send_message(message.from_user.id, "Или, Введите описание товара чтобы получить его ТН ВЭД код")

    if re.match("^\d+$", message.text):
        if message.text[0] == '0':
            message.text = message.text[1:]
        try:
            bot.send_message(message.from_user.id, f'Описание: \n{opisanie[message.text]}')
        except:
            bot.send_message(message.from_user.id, f'Описание не найдено')
        # bot.send_message(message.from_user.id, f"Это код: {message.text}")
    elif re.match("^\w+", message.text):
        description = clean(message.text)
        vector = vectorize(vectorizer, description, voc)
        kneighbors = knn.kneighbors(vector, return_distance=False)
        test_predict = knn.predict(vector)
        predicts = get_value(indexes, kneighbors)
        checks = {}
        bot.send_message(message.from_user.id, f'найдены похожие запросы')
        for i in predicts:
            checks[i] = i
        k = 0
        for i in checks:
            next_variant = f'{i}'
            try:
                next_opisanie = opisanie[next_variant]
            except:
                next_opisanie = "Описание отсутвует"
            while len(next_variant) < 4:
                next_variant = '0' + next_variant

            bot.send_message(message.from_user.id, f"{next_variant}: {next_opisanie}")
            k += 1
            if k > 4:
                break
        bot.send_message(message.from_user.id, f'Рекомендуемый код: {test_predict}')


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


def load_voc(vocabulary_file):
    with open(vocabulary_file, 'r') as f:
        voc = json.load(f)
    return voc


def load_model(model_file):
    with open(model_file, 'rb') as f:
        knn = pickle.load(f)
    return knn


def create_vectorizer(voc):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, vocabulary=voc)
    return vectorizer


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

def load_opisanie():
    with open('TNVED_OPISANIE_full.json', 'r') as f:
        opisanie = json.load(f)
    return opisanie

voc = load_voc('voc2.json')
vectorizer = create_vectorizer(voc)
indexes = load_indexes('indexes2.json')
knn = load_model('hacks_ai_top.pkl')
opisanie = load_opisanie()
bot.polling(none_stop=True, interval=0)
