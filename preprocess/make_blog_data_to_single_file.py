# -*- coding: utf-8 -*-

import csv
import os
import pickle
import re

ROWDATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../row_data/blog-articles')
PREPROCESSED_DIR = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'preprocessed_data/blog-articles')


def dump_blog_data(file_list, pickle_filename):
    blog_list = []

    for file in file_list:
        blog_dict = {}

        with open(os.path.join(ROWDATA_DIR, file), 'r') as rf:
            reader = csv.reader(rf)

            for idx, row in enumerate(reader):
                blog_dict['date'] = row[0]               # {"date": "ブログ投稿日"}
                blog_dict['title'] = row[1]              # {"title": "ブログ記事タイトル"}
                if idx == 0:
                    blog_dict['article'] = []
                    blog_dict['article'].append(row[2])  # {"article" : "ブログ本文"}
                else:
                    blog_dict['article'].append(row[2])
        blog_list.append(blog_dict)

    with open(os.path.join(PREPROCESSED_DIR, pickle_filename), 'wb') as wf:
        pickle.dump(blog_list, wf)

if __name__ == '__main__':

    files = os.listdir(ROWDATA_DIR)
    # print(files)

    pattern = 'ogurayui-0815_.*'
    ogurayui_files = [file for file in files if re.match(pattern, file)]

    pattern = 'ishiharakaori-0806_.*'
    ishiharakaori_files = [file for file in files if re.match(pattern, file)]

    dump_blog_data(ogurayui_files, 'ogurayui-blog.pkl')
    dump_blog_data(ishiharakaori_files, 'ishiharakaori-blog.pkl')
