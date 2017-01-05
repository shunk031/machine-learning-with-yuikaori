# -*- coding: utf-8 -*-

import csv
import os
import pickle
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../data/blog-articles')


def dump_blog_data(file_list, pickle_filename):
    blog_list = []

    for file in file_list:
        blog_dict = {}

        with open(os.path.join(DATA_DIR, file), 'r') as rf:
            reader = csv.reader(rf)
            for idx, row in enumerate(reader):
                if idx == 0:
                    blog_dict[row[1]] = []
                    blog_dict[row[1]].append(row[2])
                else:
                    blog_dict[row[1]].append(row[2])
        blog_list.append(blog_dict)

    with open(os.path.join(DATA_DIR, pickle_filename), 'wb') as wf:
        pickle.dump(blog_list, wf)

if __name__ == '__main__':

    files = os.listdir(DATA_DIR)
    # print(files)

    pattern = 'ogurayui-0815_.*'
    ogurayui_files = [file for file in files if re.match(pattern, file)]

    pattern = 'ishiharakaori-0806_.*'
    ishiharakaori_files = [file for file in files if re.match(pattern, file)]

    dump_blog_data(ogurayui_files, 'ogurayui-blog.pkl')
    dump_blog_data(ishiharakaori_files, 'ishiharakaori-blog.pkl')
