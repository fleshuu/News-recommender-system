import os
from flask import Flask, flash, request, redirect, url_for, render_template
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
with open("models/topic_dict.json") as json_file:
    topic_dict = json.load(json_file)
with open("models/view_topic_dict.json") as json_file:
    view_topic = json.load(json_file)
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
    return render_template('index.html', ners=ner, topicNumber=topic, words=view_topic[topic][:20])


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


if __name__ == '__main__':
    app.run(debug=True)
