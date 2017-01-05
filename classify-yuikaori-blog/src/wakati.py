# -*- coding: utf-8 -*-

import MeCab
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../data/blog-articles')


def make_wakati_list(blod_data):
    mt = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    mt.parse('')

    wakati_list = []
    for blog_dict in blog_data:
        for article_list in blog_dict.values():
            wakati_tokens = []
            for sentence in article_list:
                node = mt.parseToNode(sentence)
                while node:
                    pos = node.feature.split(",")[0]
                    if pos == "名詞":
                        # print(node.surface)
                        wakati_tokens.append(node.surface)
                    node = node.next

            wakati_list.append(' '.join(wakati_tokens))

    return wakati_list


if __name__ == '__main__':

    with open(os.path.join(DATA_DIR, 'ogurayui-blog.pkl'), 'rb') as rf:
        yui_blog_data = pickle.load(rf)

    with open(os.path.join(DATA_DIR, 'ishiharakaori-blog.pkl'), 'rb') as rf:
        kaori_blog_data = pickle.load(rf)

    yui_wakati = make_wakati_list(yui_blog_data)
    kaori_wakati = make_wakati_list(kaori_blog_data)
    X = yui_wakati + kaori_wakati
    y = ['yui'] * len(yui_wakati) + ['kaori'] * len(kaori_wakati)
