# -*- coding: utf-8 -*-

import MeCab
import pickle
import os
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib

PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../data/blog-articles')

if __name__ == '__main__':

    # ブログデータの読み込み
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, 'wakati-tokens.csv'))
    kaori_data = df[df['label'] == 'kaori']
    num_of_kaori_data = len(kaori_data)
    yui_data = df[df['label'] == 'yui']
    yui_data = yui_data.iloc[random.sample(list(yui_data.index), num_of_kaori_data)]

    df = pd.concat([yui_data, kaori_data])

    # クラスラベル'yui'と'kaori'を整数値に変更
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['label'].values)

    X = df['article'].values

    print('Split into train data and test data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # print('X train data:\n{}'.format(X_train))
    # print('X test data:\n{}'.format(X_test))

    # 学習データを用いてTF-IDFを計算
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(X_train)
    X_train_tfidf = tfidf_vec.transform(X_train)
    # print(X_train_tfidf)
    X_test_tfidf = tfidf_vec.transform(X_test)

    # Grid searchでチューニングするハイパーパラメータを設定
    svm_tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': [2**n for n in range(-15, 3)],
            'C': [2**n for n in range(-5, 15)]
        }
    ]

    # scores = ['accuracy', 'precision', 'recall']
    scores = ['accuracy']
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

    # TF-IDFモデルをdump
    print('Dump TF-IDF vectorizer model to "tfidf.pkl"')
    with open("tfidf.pkl", 'wb') as wf:
        pickle.dump(tfidf_vec, wf)
    # joblib.dump(tfidf_vec, 'tfidf.pkl', compress=1)

    # チューニングしたモデルをdump
    print('Dump best estimator model to "gridsearch_svm.pkl"')
    joblib.dump(clf.best_estimator_, 'gridsearch_svm.pkl', compress=1)
