#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:49:55 2020

@author: pasha
"""
import re
import math
import pandas as pd
from collections import Counter
import numpy as np
import copy
from sklearn.metrics import classification_report


# %%
# f_path = '/Users/pasha/PycharmProjects/Assignment2/hns_2018_2019.csv'
# df = pd.read_csv(f_path)

# #%%
# # getting post types in an array
# post_types = df["Post Type"].unique()

# # extract the year from "created at" column
# df['year_created_at'] = pd.DatetimeIndex(df['Created At']).year

# # data frame that has only 2018 as training data
# df_year_2018 = df.loc[df['year_created_at'] == 2018]

# # data frame that has only 2018 as training data
# df_year_2019 = df.loc[df['year_created_at'] == 2019]

# #%%
# # generate the vocabulary
# text_join = ' '.join(df_year_2018["Title"].values)
# text_strip = re.sub('[:.,?!;\[\]“”"\'’()]', '', text_join)
# text_strip = text_strip.lower()
# text_split = text_strip.split()
# text_split_set = set(text_split)
# text_split_set = sorted(text_split_set)
# #%%


# df_year_2018_story = df_year_2018.loc[df_year_2018['Post Type'] == 'story']
# text_join_story = ' '.join(df_year_2018_story["Title"].values)
# text_strip_story = re.sub('[:.,?!;\[\]“”"\'’()]', '', text_join_story)
# text_strip_story = text_strip_story.lower()
# text_split_story = text_strip_story.split()

# #%%
# text_split_dict = {x:0.5 for x in text_split_set}
# # text_split_set = dict(Counter())
# total = sum(text_split_dict.values())
# text_prob_dict = {x: text_split_dict[x]/total for x in text_split_dict.keys()}
# # dict_merged2 = {x:text_split_dict[x]+cnt[x] for x in text_split_dict.keys()}

# cnt = Counter(text_split_story)
# dict_merged = {x:text_split_dict.get(x,0)+cnt.get(x,0) for x in text_split_dict.keys()}
# #%%

def strip_char(text):
    with open('remove_word.txt', 'a') as f:
        text_strip = text
        # if not text.find("^\d") == -1:
        #     f.write('%s\n'%"numbers")
        #     text_strip = re.sub('/^\d/', '', text_strip)
        if not text.find(":") == -1:
            f.write('%s\n' % ":")
            text_strip = re.sub('[:]', '', text_strip)
        if not text.find(".") == -1:
            f.write('%s\n' % ".")
            text_strip = re.sub('[.]', '', text_strip)
        if not text.find(",") == -1:
            f.write('%s\n' % ",")
            text_strip = re.sub('[,]', '', text_strip)
        if not text.find("?") == -1:
            f.write('%s\n' % "?")
            text_strip = re.sub('[?]', '', text_strip)
        if not text.find("!") == -1:
            f.write('%s\n' % "!")
            text_strip = re.sub('[!]', '', text_strip)
        if not text.find(";") == -1:
            f.write('%s\n' % ";")
            text_strip = re.sub('[;]', '', text_strip)
        if not text.find("[") == -1:
            f.write('%s\n' % "[")
            text_strip = re.sub('[\[]', '', text_strip)
        if not text.find("]") == -1:
            f.write('%s\n' % "]")
            text_strip = re.sub('[\]]', '', text_strip)
        if not text.find("“") == -1:
            f.write('%s\n' % "“")
            text_strip = re.sub('[“]', '', text_strip)
        if not text.find("”") == -1:
            f.write('%s\n' % "”")
            text_strip = re.sub('[”]', '', text_strip)
        if not text.find("\"") == -1:
            f.write('%s\n' % "\"")
            text_strip = re.sub('[\"]', '', text_strip)
        if not text.find("\'") == -1:
            f.write('%s\n' % "\'")
            text_strip = re.sub('[\']', '', text_strip)
        if not text.find("’") == -1:
            f.write('%s\n' % "’")
            text_strip = re.sub('[’]', '', text_strip)
        if not text.find("(") == -1:
            f.write('%s\n' % "(")
            text_strip = re.sub('[(]', '', text_strip)
        if not text.find(")") == -1:
            f.write('%s\n' % ")")
            text_strip = re.sub('[)]', '', text_strip)
        if not text.find("‘") == -1:
            f.write('%s\n' % "‘")
            text_strip = re.sub('[‘]', '', text_strip)
    return text_strip


def vocabulary(text):
    with open('vocabulary.txt', 'a') as f:
        #
        text = text.lower()
        #
        vocabulary = text.split()
        #
        vocabulary = set(vocabulary)
        #
        vocabulary = sorted(vocabulary)
        #
        for elem in vocabulary:
            f.write('%s\n' % elem)
        # make a dictionary out of vocabulary set and set the keys to 0.5 as smoothing
        vocab_dict = {x: 0.5 for x in vocabulary}
    return vocab_dict


def freq_prob_types(vocab_dict, df, year, file_name):
    # # make a dictionary out of vocabulary set and set the keys to 0.5 as smoothing
    # vocab_dict = {x:0.5 for x in vocab}

    # extract the year from "created at" column
    df['year_created_at'] = pd.DatetimeIndex(df['Created At']).year

    # data frame that has only the year passed to the function as the training data
    df_training = df.loc[df['year_created_at'] == year]

    # getting post types in an array
    post_types = df_training["Post Type"].unique()
    # post_types_freq = df_training["Post Type"].unique()
    # post_types_prob = df_training["Post Type"].unique()

    post_types_stats = dict()
    for elem in post_types:
        df_temp = df_training.loc[df_training['Post Type'] == elem]
        number_of_elements = len(df_temp)
        text_temp = ' '.join(df_temp["Title"].values)
        text_temp = re.sub('[:.,?!;\[\]“”"\'’()]', '', text_temp)
        text_temp = text_temp.lower()
        text_temp_split = text_temp.split()
        cnt = Counter(text_temp_split)
        post_types_freq = {x: vocab_dict.get(x, 0) + cnt.get(x, 0) for x in vocab_dict.keys()}

        total = sum(post_types_freq.values())
        post_types_prob = {x: post_types_freq[x] / total for x in post_types_freq.keys()}
        post_types_stats[elem] = [post_types_freq, post_types_prob, number_of_elements]

    with open('model-2018.txt', 'a') as f:
        for (i, word) in enumerate(vocab_dict):
            f.write('%d  %s  ' % (i + 1, word))
            for post_type in post_types:
                f.write('%s: %d  %f  ' % (
                post_type, post_types_stats[post_type][0][word], post_types_stats[post_type][1][word]))

            # for k in range(len(post_types)):
            #     f.write('%s: %d  %f  '%(post_types[k], post_types_freq[k][j], post_types_prob[k][j]))
            f.write("\n")

    return post_types_stats

    # create dictionary for each post type for frequencies

    # create dictionary for each post type for probabilities


def classifier(string, post_types_stats):
    # string = strip_char(string)
    string = re.sub('[:.,?!;\[\]“”"\'’()]', '', string)
    string = string.lower()
    words_list = string.split()
    total_titles = sum([post_types_stats[post_type][2] for post_type in post_types_stats.keys()])

    scores = dict()

    for post_type in post_types_stats.keys():
        post_types_freq, post_types_prob, number_of_elements = post_types_stats[post_type]
        vocab = post_types_prob.keys()
        score = np.log10(number_of_elements / total_titles)
        for word in words_list:
            if word in vocab:
                score += np.log10(post_types_prob.get(word, 0))
        scores[post_type] = score

    return scores


def get_answer_from_scores(string, post_types_stats):
    scores = classifier(string, post_types_stats)
    return max(scores, key=scores.get)


def write_classification_to_file(string, post_types_stats):
    scores = classifier(string, post_types_stats)

    return


if __name__ == "__main__":
    f_path = '/Users/pasha/PycharmProjects/Assignment2/hns_2018_2019.csv'
    df = pd.read_csv(f_path)

    # extract the year from "created at" column
    df['year_created_at'] = pd.DatetimeIndex(df['Created At']).year

    # data frame that has only 2018 as training data
    df_year_2018 = df.loc[df['year_created_at'] == 2018]

    # data frame that has only 2019 as training data
    df_year_2019 = df.loc[df['year_created_at'] == 2019]

    text_join = ' '.join(df_year_2018["Title"].values)

    text_s = strip_char(text_join)

    voc = vocabulary(text_s)

    post_types_stats = freq_prob_types(voc, df, 2018, 'model-2018.txt')

    corrects = 0

    with open('baseline-result.txt', 'a') as f:
        for (i, row) in enumerate(df_year_2019.iterrows()):
            _, row = row
            title = row['Title']
            scores = classifier(title, post_types_stats)
            prediction = max(scores, key=scores.get)
            result = 'right' if prediction == row['Post Type'] else 'wrong'
            if result == 'right':
                corrects += 1
            f.write('%d  %s  %s  ' % (i + 1, title, prediction))
            for key in scores:
                f.write('%s:  %f  ' % (key, scores[key]))
            f.write('%s  %s\n' % (row['Post Type'], result))
    print(corrects)

    s_path = '/Users/pasha/PycharmProjects/Assignment2/stopwords.txt'

    # with open(s_path, 'r') as s:
    #    l=[]
    l = [line.rstrip('\n') for line in open(s_path)]
    # for line in s:
    #     l.append(line)
    voc_stop_words = copy.deepcopy(voc)
    for elem in l:
        voc_stop_words.pop(elem, None)
    post_types_stats = freq_prob_types(voc_stop_words, df, 2018, 'stopword-model.txt')

    corrects = 0

    with open('stopword-result.txt', 'a') as f:
        for (i, row) in enumerate(df_year_2019.iterrows()):
            _, row = row
            title = row['Title']
            scores = classifier(title, post_types_stats)
            prediction = max(scores, key=scores.get)
            result = 'right' if prediction == row['Post Type'] else 'wrong'
            if result == 'right':
                corrects += 1
            f.write('%d  %s  %s  ' % (i + 1, title, prediction))
            for key in scores:
                f.write('%s:  %f  ' % (key, scores[key]))
            f.write('%s  %s\n' % (row['Post Type'], result))
    print(corrects)
