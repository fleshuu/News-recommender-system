import datetime
from flask import Flask, request, render_template
from recommender.TextProcessing import TextProcessing
from deeppavlov import configs, build_model
import json
from bs4 import BeautifulSoup
import requests

app = Flask('Recommender system')

tp = TextProcessing()
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
with open("models/ner_dict.json") as json_file:
    ner_dict = json.load(json_file)
with open("models/topic_url_dict.json") as json_file:
    topic_url_dict = json.load(json_file)
with open("models/topic_dict.json") as json_file:
    topic_dict = json.load(json_file)
with open("models/view_topic_dict.json") as json_file:
    view_topic = json.load(json_file)
with open("models/url_title_dict.json") as json_file:
    url_title_dict = json.load(json_file)
today = datetime.date.today()
symbols = ['.', '!', '?', '।', '።', '။', '。', ',', ':', ';', '؟', '،', \
           ' ', '+', '-', '*', '/', '"', "'", '{', '}', '[', ']', '(', ')', '“']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    url_text = request.form['option']
    if url_text == 'text':
        text = request.form['sent']
    else:
        URL = request.form['sent']
        response = requests.get(URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        p = []
        for i in soup.select('p'):
            if i.string is not None:
                c = tp.cleaner(i.string)
                if not c.isspace():
                    p += [c]
        title = soup.title.string
        text = title * 2
        text += tp.from_list_to_texts(p)

    ner, topic = ner_topic(text)
    print(ner, topic)
    recommendations = get_recommendations(ner, topic)
    ordered_recommendations = order_recommendations(recommendations)
    print(ordered_recommendations)
    return render_template('index.html', ners=ner, topicNumber=topic,
                           words=view_topic[topic][:20], recommendation=ordered_recommendations)


def ner_topic(text):
    chunkedText = tp.chunker(text)
    ner = []
    if chunkedText:
        doc = ner_model(chunkedText)
        for j in range(len(doc[0])):
            for i in range(len(doc[0][j])):
                token = doc[0][j][i]
                tag = doc[1][j][i]
                if tag != 'O':
                    if tag[2:] not in ['TIME', 'DATE', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                        ner += [[token.lower(), tag]]
        ner = tp.tagUniter(ner)
        text = get_words(text)
        topicN = get_topic(text)
        return ner, topicN


def get_words(text, symbols=symbols):
    word = ''
    tokens = []
    for char in text:
        if word is not '' and char in symbols:
            tokens += [word.lower()]
            word = ''
        if char not in symbols:
            word += char
    if word is not '':
        tokens += [word.lower()]
    return tokens


def get_topic(list_text):
    topics = {}
    for word in list_text:
        if topic_dict.get(word):
            for i in topic_dict.get(word):
                topicN = i[0]
                topicSc = i[1]
                if topics.get(topicN):
                    topics[topicN] += topicSc
                else:
                    topics[topicN] = topicSc
    sorted_d = {k: v for k, v in sorted(topics.items(), key=lambda item: item[1], reverse=True)}
    topic = None
    for i in sorted_d.keys():
        topic = i
        break

    return topic


def get_recommendations(ner_words, topic):
    url_score = {}
    for word, tag in ner_words:
        if ner_dict.get(word):
            for url, df in ner_dict.get(word):
                if url_score.get(url):
                    url_score[url] += df
                else:
                    url_score[url] = df
    if topic_url_dict.get(topic):
        for url in topic_url_dict.get(topic):
            if url_score.get(url):
                url_score[url] += 2
            else:
                url_score[url] = 2
    return url_score


def order_recommendations(url_score):
    ordered = {}
    for url, score in url_score.items():
        url_title = url_title_dict.get(url)
        if url_title:
            create_date = url_title.get('created_date')
            title = url_title['title']
            year = int(create_date[:4])
            month = int(create_date[5:7])
            day = int(create_date[8:10])
            date_of_url = datetime.date(year, month, day)
            diff = today - date_of_url
            diff_days = diff.days
            if diff_days < 0:
                print(url, date_of_url)
                diff_days = 100
            if diff_days > 370:
                diff_days = 370 + diff_days // 10
            shift_diff_days = diff_days + 1.1
            multiplier = round(1/shift_diff_days, 4)
            ordered[url] = [score * multiplier, title]
    ordered = [[k, v] for k, v in sorted(ordered.items(), key=lambda item: item[1], reverse=True)]

    return ordered[:10]


if __name__ == '__main__':
    app.run(debug=True)
