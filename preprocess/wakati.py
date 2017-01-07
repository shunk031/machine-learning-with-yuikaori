# -*- coding: utf-8 -*-

import MeCab
import pickle
import os
import csv


PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'preprocessed_data/blog-articles')


def make_wakati_list(blog_data):

    wakati_list = [make_wakati_string(blog_dict['article']) for blog_dict in blog_data]

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

    with open(os.path.join(PREPROCESSED_DIR, 'ogurayui-blog.pkl'), 'rb') as rf:
        yui_blog_data = pickle.load(rf)

    with open(os.path.join(PREPROCESSED_DIR, 'ishiharakaori-blog.pkl'), 'rb') as rf:
        kaori_blog_data = pickle.load(rf)

    yui_wakati_list = make_wakati_list(yui_blog_data)
    kaori_wakati_list = make_wakati_list(kaori_blog_data)

    with open(os.path.join(PREPROCESSED_DIR, 'wakati-tokens.csv'), 'w') as wf:
        writer = csv.writer(wf)

        header = ['date', 'title', 'article', 'classlabel']
        writer.writerow(header)

        for wakati_string, blog_dict in zip(yui_wakati_list, yui_blog_data):
            writer.writerow([blog_dict['date'], blog_dict['title'], [wakati_string], 'yui'])

        for wakati_string, blog_dict in zip(kaori_wakati_list, kaori_blog_data):
            writer.writerow([blog_dict['date'], blog_dict['title'], [wakati_string], 'kaori'])
