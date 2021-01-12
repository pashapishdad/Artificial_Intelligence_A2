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
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


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


def freq_prob_types(vocab_dict, df, year, file_name=""):
    # extract the year from "created at" column
    df['year_created_at'] = pd.DatetimeIndex(df['Created At']).year

    # data frame that has only the year passed to the function as the training data
    df_training = df.loc[df['year_created_at'] == year]

    # getting post types in an array
    global post_types
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
    if not file_name == "":
        with open(file_name, 'a') as f:
            for (i, word) in enumerate(vocab_dict):
                f.write('%d  %s  ' % (i + 1, word))
                for post_type in post_types:
                    f.write('%s: %d  %f  ' % (
                    post_type, post_types_stats[post_type][0][word], post_types_stats[post_type][1][word]))
                f.write("\n")

    return post_types_stats


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


def vocab_freqency(voc, freq):
    voc_words_frequency = copy.deepcopy(voc)
    l_freq = []
    for word in voc.keys():
        sum_freq = 0
        for post_type in post_types_stats.keys():
            sum_freq += (post_types_stats[post_type][0][word] - 0.5)
        if sum_freq <= freq:
            l_freq.append(word)
    for elem in l_freq:
        voc_words_frequency.pop(elem, None)
    return voc_words_frequency


def vocab_frequent(voc, percent):
    voc_words_frequent = copy.deepcopy(voc)
    l_freq = [k for k, _ in sorted(voc.items(), key=lambda item: item[1], reverse=True)]
    for i in range(int(len(l_freq) * percent / 100)):
        l_freq.pop()
    for elem in l_freq:
        voc_words_frequent.pop(elem, None)
    return voc_words_frequent


def performance(post_types_stats, file_name=""):
    if not file_name == "":
        corrects = 0

        with open(file_name, 'a') as f:
            predictions = []
            trues = []
            for (i, row) in enumerate(df_year_2019.iterrows()):
                _, row = row
                title = row['Title']
                scores = classifier(title, post_types_stats)
                prediction = max(scores, key=scores.get)
                predictions.append(prediction)
                trues.append(row['Post Type'])
                result = 'right' if prediction == row['Post Type'] else 'wrong'
                if result == 'right':
                    corrects += 1
                f.write('%d  %s  %s  ' % (i + 1, title, prediction))
                for key in scores:
                    f.write('%s:  %f  ' % (key, scores[key]))
                f.write('%s  %s\n' % (row['Post Type'], result))
        print(corrects)
    else:
        corrects = 0
        predictions = []
        trues = []
        for (i, row) in enumerate(df_year_2019.iterrows()):
            _, row = row
            title = row['Title']
            scores = classifier(title, post_types_stats)
            prediction = max(scores, key=scores.get)
            predictions.append(prediction)
            trues.append(row['Post Type'])
            result = 'right' if prediction == row['Post Type'] else 'wrong'
            if result == 'right':
                corrects += 1
        print(corrects)
        report = classification_report(trues, predictions, target_names=post_types, output_dict=True)
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1score = report['weighted avg']['f1-score']
        return accuracy, precision, recall, f1score, corrects


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

    performance(post_types_stats, 'baseline-result.txt')

    # with open('baseline-result.txt', 'a') as f:
    #     for (i,row) in enumerate(df_year_2019.iterrows()):
    #         _ ,row = row
    #         title = row['Title']
    #         scores = classifier(title, post_types_stats)
    #         prediction = max(scores,key=scores.get)
    #         result = 'right' if prediction == row['Post Type'] else 'wrong'
    #         if result == 'right':
    #             corrects += 1
    #         f.write('%d  %s  %s  '%(i + 1, title, prediction))
    #         for key in scores:
    #             f.write('%s:  %f  '%(key, scores[key]))
    #         f.write('%s  %s\n'%(row['Post Type'],result))
    # print(corrects)

    s_path = '/Users/pasha/PycharmProjects/Assignment2/stopwords.txt'
    l = [line.rstrip('\n') for line in open(s_path)]
    voc_stop_words = copy.deepcopy(voc)
    for elem in l:
        voc_stop_words.pop(elem, None)
    post_types_stats2 = freq_prob_types(voc_stop_words, df, 2018, 'stopword-model.txt')
    performance(post_types_stats2, 'stopword-result.txt')

    # corrects = 0

    # with open('stopword-result.txt', 'a') as f:
    #     predictions = []
    #     trues = []
    #     for (i,row) in enumerate(df_year_2019.iterrows()):
    #         _ ,row = row
    #         title = row['Title']
    #         scores = classifier(title, post_types_stats2)
    #         prediction = max(scores,key=scores.get)
    #         predictions.append(prediction)
    #         trues.append(row['Post Type'])
    #         result = 'right' if prediction == row['Post Type'] else 'wrong'
    #         if result == 'right':
    #             corrects += 1
    #         f.write('%d  %s  %s  '%(i + 1, title, prediction))
    #         for key in scores:
    #             f.write('%s:  %f  '%(key, scores[key]))
    #         f.write('%s  %s\n'%(row['Post Type'],result))
    # print(corrects)

    # report = classification_report(trues, predictions, target_names=post_types, output_dict=True)
    # report2 = precision_recall_fscore_support(trues, predictions, average=None, labels=post_types)
    # print(report)

    voc_words_length = copy.deepcopy(voc)
    l_word_length = []
    for key in voc_words_length.keys():
        if not 2 <= len(key) <= 9:
            l_word_length.append(key)
    for elem in l_word_length:
        voc_words_length.pop(elem, None)

    post_types_stats3 = freq_prob_types(voc_words_length, df, 2018, 'wordlength-model.txt')

    corrects = 0
    with open('wordlength-result.txt', 'a') as f:
        predictions = []
        trues = []
        for (i, row) in enumerate(df_year_2019.iterrows()):
            _, row = row
            title = row['Title']
            scores = classifier(title, post_types_stats3)
            prediction = max(scores, key=scores.get)
            predictions.append(prediction)
            trues.append(row['Post Type'])
            result = 'right' if prediction == row['Post Type'] else 'wrong'
            if result == 'right':
                corrects += 1
            f.write('%d  %s  %s  ' % (i + 1, title, prediction))
            for key in scores:
                f.write('%s:  %f  ' % (key, scores[key]))
            f.write('%s  %s\n' % (row['Post Type'], result))
    print(corrects)

    # voc_words_freq = copy.deepcopy(voc)
    # l_freq = []
    # for word in voc.keys():
    #     sum_freq = 0
    #     for post_type in post_types_stats.keys():
    #         sum_freq += (post_types_stats[post_type][0][word] - 0.5)
    #         # if (post_types_stats[post_type][0][word] - 0.5) > max_freq:
    #         #     max_freq = post_types_stats[post_type][0][word] - 0.5
    #     if sum_freq <= freq:
    #         l_freq.append(word)

    # for elem in l_freq:
    #     voc_words_freq.pop(elem, None)

    voc_words_frequency = vocab_freqency(voc, 1)
    post_types_stats4 = freq_prob_types(voc_words_frequency, df, 2018)

    corrects = 0
    # with open('wordlength-result.txt', 'a') as f:
    predictions = []
    trues = []
    for (i, row) in enumerate(df_year_2019.iterrows()):
        _, row = row
        title = row['Title']
        scores = classifier(title, post_types_stats4)
        prediction = max(scores, key=scores.get)
        predictions.append(prediction)
        trues.append(row['Post Type'])
        result = 'right' if prediction == row['Post Type'] else 'wrong'
        if result == 'right':
            corrects += 1
        # f.write('%d  %s  %s  '%(i + 1, title, prediction))
        # for key in scores:
        #     f.write('%s:  %f  '%(key, scores[key]))
        # f.write('%s  %s\n'%(row['Post Type'],result))
    print(corrects)

    voc_words_frequent = vocab_frequent(voc, 15)
    post_types_stats5 = freq_prob_types(voc_words_frequent, df, 2018)

    corrects = 0
    predictions = []
    trues = []
    for (i, row) in enumerate(df_year_2019.iterrows()):
        _, row = row
        title = row['Title']
        scores = classifier(title, post_types_stats5)
        prediction = max(scores, key=scores.get)
        predictions.append(prediction)
        trues.append(row['Post Type'])
        result = 'right' if prediction == row['Post Type'] else 'wrong'
        if result == 'right':
            corrects += 1
    print(corrects)
    # x = [1,2,3,4,5]
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(x, [report['accuracy']]*5, marker = '*', label='accuracy')
    # plt.subplot(1,2,2)
    # plt.plot(x, [report['accuracy']]*5, marker = '*', label='accuracy')
    # plt.legend(loc='center left', bbox_to_anchor=(1,.5))
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(x, [report['accuracy']]*5, marker = '*', label='accuracy')
    # plt.subplot(1,2,2)
    # plt.plot(x, [report['accuracy']]*5, marker = '*', label='accuracy')
    # plt.legend(loc='center left', bbox_to_anchor=(1,.5))
    # plt.tight_layout()
    # plt.subplot(1,2,1)
    # plt.xticks(ticks=x)
    # list_of_reports = report*5
    # list_of_reports = [report]*5

    # accuracies = [rep['accuracy'] for rep in list_of_reports]

    # precisions = [rep['weighted avg']['precision'] for rep in list_of_reports]

    # recalls = [rep['weighted avg']['recall'] for rep in list_of_reports]

    # f1scores = [rep['weighted avg']['f1-score'] for rep in list_of_reports]
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(x, accuracies, marker = '*', label='accuracy')
    # plt.plot(x, precisions, marker = 'o', label='precision')