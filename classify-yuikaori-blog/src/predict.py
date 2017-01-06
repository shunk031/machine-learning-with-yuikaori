# -*- coding: utf-8 -*-

import csv
import os
import MeCab

from sklearn.externals import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../data/test')


if __name__ == '__main__':

    # testdata_filename = "ogurayui-0815_2017-01-05_AKIBA'S TRIP☆.csv"
    testdata_filename = "ishiharakaori-0806_2017-01-06_ステキ☆.csv"

    # テストデータの読み込み
    with open(os.path.join(DATA_DIR, testdata_filename), 'r') as rf:
        reader = csv.reader(rf)
        blog_sentences = [row[2] for row in reader]

    # print(blog_sentences)

    mt = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    mt.parse('')

    wakati_tokens = []
    for sentence in blog_sentences:
        node = mt.parseToNode(sentence)

        while node:
            pos = node.feature.split(',')[0]
            if pos == '名詞':
                wakati_tokens.append(node.surface)
            node = node.next

    wakati_tokens = [' '.join(wakati_tokens)]
    print(wakati_tokens)

    # Trainで利用したTF-IDFモデルの読み込み
    tfidf_vec = joblib.load("tfidf.pkl")
    X_test_tfidf = tfidf_vec.transform(wakati_tokens)
    print('X test tfidf:\n{}'.format(X_test_tfidf))

    # dumpしたSVMモデルの読み込み
    svm_est = joblib.load("gridsearch_svm.pkl")

    # テストデータの予測
    pred = svm_est.predict(X_test_tfidf)

    if pred == 0:
        print("Predict: input article is ISHIHARA Kaori's blog.")
    else:
        print("Predict: input article is OGURA Yui's blog.")
