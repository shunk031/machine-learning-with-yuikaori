# -*- coding: utf-8 -*-

import MeCab
import pickle
import os
import csv


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../data/blog-articles')


def make_wakati_list(blog_data):
    wakati_list = []
    for blog_dict in blog_data:
        for article_list in blog_dict.values():
            wakati_string = make_wakati_string(article_list)
            wakati_list.append(wakati_string)
    return wakati_list


def make_wakati_string(article_list):

    mt = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    mt.parse('')

    wakati_tokens = []
    for sentence in article_list:
        node = mt.parseToNode(sentence)
        while node:
            pos = node.feature.split(',')[0]
            if pos == '名詞':
                wakati_tokens.append(node.surface)
            node = node.next

    wakati_string = ' '.join(wakati_tokens)
    return wakati_string

if __name__ == '__main__':

    with open(os.path.join(DATA_DIR, 'ogurayui-blog.pkl'), 'rb') as rf:
        yui_blog_data = pickle.load(rf)

    with open(os.path.join(DATA_DIR, 'ishiharakaori-blog.pkl'), 'rb') as rf:
        kaori_blog_data = pickle.load(rf)

    yui_wakati_list = make_wakati_list(yui_blog_data)
    kaori_wakati_list = make_wakati_list(kaori_blog_data)

    with open(os.path.join(DATA_DIR, 'wakati-tokens.csv'), 'w') as wf:
        writer = csv.writer(wf)

        header = ['label', 'article']
        writer.writerow(header)

        for wakati_string in yui_wakati_list:
            # print(wakati_string)
            writer.writerow(["yui", [wakati_string]])

        for wakati_string in kaori_wakati_list:
            # print(wakati_string)
            writer.writerow(["kaori", [wakati_string]])
