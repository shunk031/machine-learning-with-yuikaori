# -*- coding: utf-8 -*-

import MeCab
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../data/blog-articles')


def make_wakati_list(blog_data):
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

    # ブログ記事を分かち書き
    print('Now wakatigaking...')
    yui_wakati = make_wakati_list(yui_blog_data)
    kaori_wakati = make_wakati_list(kaori_blog_data)

    X = yui_wakati + kaori_wakati
    y = ['yui'] * len(yui_wakati) + ['kaori'] * len(kaori_wakati)

    # クラスラベル'yui'と'kaori'を整数値に変更
    class_le = LabelEncoder()
    y = class_le.fit_transform(y)

    print('split into train data and test data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 学習データを用いてTF-IDFを計算
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(X_train)
    X_train_tfidf = tfidf_vec.transform(X_train)
    X_test_tfidf = tfidf_vec.transform(X_test)

    # Grid searchでチューニングするハイパーパラメータを設定
    svm_tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': [2**n for n in range(-15, 3)],
            'C': [2**n for n in range(-5, 15)]
        }
    ]

    scores = ['accuracy', 'precision', 'recall']
    for score in scores:
        print('\n' + '=' * 50)
        print(score)
        print('=' * 50)

        clf = GridSearchCV(svm.SVC(), svm_tuned_parameters, cv=5, scoring=score, n_jobs=-1)
        clf.fit(X_train_tfidf, y_train)

        print("\n+ ベストパラメータ:\n")
        print(clf.best_estimator_)

        print("\n+ トレーニングデータでCVした時の平均スコア:\n")
        for params, mean_score, all_scores in clf.grid_scores_:
            print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

        print("\n+ テストデータでの識別結果:\n")
        y_true, y_pred = y_test, clf.predict(X_test_tfidf)
        print(classification_report(y_true, y_pred))
