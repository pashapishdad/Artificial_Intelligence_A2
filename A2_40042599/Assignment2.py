# ----------------------------------
# Assignment 2
# written by Pasha Pishdad 40042599
# For COMP 472 section ABKX - Summer 2020
# ----------------------------------

import re
import pandas as pd
from collections import Counter
import numpy as np
import copy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# this function will check punctuations in a text and remove it and add it to a text file if exists.
def strip_char(text):
    with open('remove_word.txt', 'a') as f:
        text_strip = text
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


# create sorted vocabulary and add 0.5 for the smoothing that will be used in other functions.
# and add it to vocabulary.text.
def vocabulary(text):
    with open('vocabulary.txt', 'a') as f:
        text = text.lower()
        vocabulary = text.split()
        vocabulary = set(vocabulary)
        vocabulary = sorted(vocabulary)
        for elem in vocabulary:
            f.write('%s\n' % elem)
        # make a dictionary out of vocabulary set and set the keys to 0.5 as smoothing
        vocab_dict = {x: 0.5 for x in vocabulary}
    return vocab_dict


# this function creates a dictionary of "Post Types" dictionaries with frequency and probabilities
# of each word in the vocabulary in each post types.
def freq_prob_types(vocab_dict, df, year, file_name=""):
    # extract the year from "created at" column
    df['year_created_at'] = pd.DatetimeIndex(df['Created At']).year
    # data frame that has only the year passed to the function as the training data
    df_training = df.loc[df['year_created_at'] == year]
    # getting post types in an array
    global post_types
    post_types = df_training["Post Type"].unique()
    # calculate frequency and probability of each word.
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
    # if a file_name will be given as an argument, it will create the model as a text file.
    if not file_name == "":
        with open(file_name, 'a') as f:
            for (i, word) in enumerate(vocab_dict):
                f.write('%d  %s  ' % (i + 1, word))
                for post_type in post_types:
                    f.write('%s: %d  %f  ' % (
                    post_type, post_types_stats[post_type][0][word], post_types_stats[post_type][1][word]))
                f.write("\n")
    return post_types_stats


# creates an array of scores which has score of each post types for a given string(title).
def classifier(string, post_types_stats):
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


# create a vocabulary for given frequency of words that should be removed from the vocabulary.
def vocab_frequency(voc, freq):
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


# create a vocabulary for given percent of top frequent words that should be removed from the vocabulary.
def vocab_frequent(voc, percent):
    voc_words_frequent = copy.deepcopy(voc)
    l_freq = [k for k, _ in sorted(voc.items(), key=lambda item: item[1], reverse=True)]
    l_removed = []
    for i in range(int(len(l_freq) * percent / 100)):
        l_removed.append(l_freq.pop())
    for elem in l_removed:
        voc_words_frequent.pop(elem, None)
    return voc_words_frequent


# this function will use the classifier function and use the max as the most probable answer
# and will write to a file if file_name is given.
def performance(post_types_stats, file_name=""):
    if not file_name == "":
        corrects = 0
        with open(file_name, 'a') as f:
            # two arrays to be used for performance, true result and prediction using the classifier.
            predictions = []
            trues = []
            # write the result to a given file_name.
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
        # create a performance report(accuracy, precision, recall, f1score)
        report = classification_report(trues, predictions, target_names=post_types, output_dict=True)
        accuracy = report['accuracy']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1score = report['weighted avg']['f1-score']
        return accuracy, precision, recall, f1score, corrects


# main function
if __name__ == "__main__":
    # start message
    print("program is running, please wait for the files to be created and for the graph to be displayed.")

    # read the file using pandas and creates the data frame.
    f_path = 'hns_2018_2019.csv'
    df = pd.read_csv(f_path)

    # extract the year from "created at" column
    df['year_created_at'] = pd.DatetimeIndex(df['Created At']).year

    # data frame that has only 2018 as training data
    df_year_2018 = df.loc[df['year_created_at'] == 2018]

    # data frame that has only 2019 as testing data
    df_year_2019 = df.loc[df['year_created_at'] == 2019]

    # create the initial vocabulary by combining all the 2018 titles as an input for vocabulary function.
    text_join = ' '.join(df_year_2018["Title"].values)
    text_s = strip_char(text_join)
    voc = vocabulary(text_s)

    # create the model for task 1.
    post_types_stats = freq_prob_types(voc, df, 2018, 'model-2018.txt')
    # baseline result
    performance(post_types_stats, 'baseline-result.txt')

    # stop word model and result.
    s_path = 'stopwords.txt'
    # removing the words from the vocabulary based on stopword file.
    l = [line.rstrip('\n') for line in open(s_path)]
    voc_stop_words = copy.deepcopy(voc)
    for elem in l:
        voc_stop_words.pop(elem, None)
    # create the model and result.
    post_types_stats2 = freq_prob_types(voc_stop_words, df, 2018, 'stopword-model.txt')
    performance(post_types_stats2, 'stopword-result.txt')

    # word length model and result.
    voc_words_length = copy.deepcopy(voc)
    # removing the words from the vocabulary based on the length of the word which should be between 2 and 9.
    l_word_length = []
    for key in voc_words_length.keys():
        if not 2 <= len(key) <= 9:
            l_word_length.append(key)
    for elem in l_word_length:
        voc_words_length.pop(elem, None)
    # create the model and result.
    post_types_stats3 = freq_prob_types(voc_words_length, df, 2018, 'wordlength-model.txt')
    performance(post_types_stats3, 'wordlength-result.txt')

    # performance metrics for the baseline
    accuracy_base, precision_base, recall_base, f1score_base, corrects_base = performance(post_types_stats)

    # performance metrics for frequency = 1
    voc_words_frequency_1 = vocab_frequency(voc, 1)
    post_types_stats_f_1 = freq_prob_types(voc_words_frequency_1, df, 2018)
    accuracy_f_1, precision_f_1, recall_f_1, f1score_f_1, corrects_f_1 = performance(post_types_stats_f_1)

    # performance metrics for frequency <= 5
    voc_words_frequency_5 = vocab_frequency(voc, 5)
    post_types_stats_f_5 = freq_prob_types(voc_words_frequency_5, df, 2018)
    accuracy_f_5, precision_f_5, recall_f_5, f1score_f_5, corrects_f_5 = performance(post_types_stats_f_5)

    # performance metrics for frequency <= 10
    voc_words_frequency_10 = vocab_frequency(voc, 10)
    post_types_stats_f_10 = freq_prob_types(voc_words_frequency_10, df, 2018)
    accuracy_f_10, precision_f_10, recall_f_10, f1score_f_10, corrects_f_10 = performance(post_types_stats_f_10)

    # performance metrics for frequency <= 15
    voc_words_frequency_15 = vocab_frequency(voc, 15)
    post_types_stats_f_15 = freq_prob_types(voc_words_frequency_15, df, 2018)
    accuracy_f_15, precision_f_15, recall_f_15, f1score_f_15, corrects_f_15 = performance(post_types_stats_f_15)

    # performance metrics for frequency <= 20
    voc_words_frequency_20 = vocab_frequency(voc, 20)
    post_types_stats_f_20 = freq_prob_types(voc_words_frequency_20, df, 2018)
    accuracy_f_20, precision_f_20, recall_f_20, f1score_f_20, corrects_f_20 = performance(post_types_stats_f_20)

    # performance metrics for frequent <= 5
    voc_words_frequent_5 = vocab_frequent(voc, 5)
    post_types_stats_t_5 = freq_prob_types(voc_words_frequent_5, df, 2018)
    accuracy_t_5, precision_t_5, recall_t_5, f1score_t_5, corrects_t_5 = performance(post_types_stats_t_5)

    # performance metrics for frequent <= 10
    voc_words_frequent_10 = vocab_frequent(voc, 10)
    post_types_stats_t_10 = freq_prob_types(voc_words_frequent_10, df, 2018)
    accuracy_t_10, precision_t_10, recall_t_10, f1score_t_10, corrects_t_10 = performance(post_types_stats_t_10)

    # performance metrics for frequent <= 15
    voc_words_frequent_15 = vocab_frequent(voc, 15)
    post_types_stats_t_15 = freq_prob_types(voc_words_frequent_15, df, 2018)
    accuracy_t_15, precision_t_15, recall_t_15, f1score_t_15, corrects_t_15 = performance(post_types_stats_t_15)

    # performance metrics for frequent <= 20
    voc_words_frequent_20 = vocab_frequent(voc, 20)
    post_types_stats_t_20 = freq_prob_types(voc_words_frequent_20, df, 2018)
    accuracy_t_20, precision_t_20, recall_t_20, f1score_t_20, corrects_t_20 = performance(post_types_stats_t_20)

    # performance metrics for frequent <= 25
    voc_words_frequent_25 = vocab_frequent(voc, 25)
    post_types_stats_t_25 = freq_prob_types(voc_words_frequent_25, df, 2018)
    accuracy_t_25, precision_t_25, recall_t_25, f1score_t_25, corrects_t_25 = performance(post_types_stats_t_25)

    # accuracies for frequency
    accuracies_frequency = [accuracy_base, accuracy_f_1, accuracy_f_5, accuracy_f_10, accuracy_f_15, accuracy_f_20]

    # precisions for frequency
    precisions_frequency = [precision_base, precision_f_1, precision_f_5, precision_f_10, precision_f_15,
                            precision_f_20]

    # recalls for frequency
    recalls_frequency = [recall_base, recall_f_1, recall_f_5, recall_f_10, recall_f_15, recall_f_20]

    # f1scores for frequency
    f1scores_frequency = [f1score_base, f1score_f_1, f1score_f_5, f1score_f_10, f1score_f_15, f1score_f_20]

    # accuracies for top frequent
    accuracies_frequent = [accuracy_base, accuracy_t_5, accuracy_t_10, accuracy_t_15, accuracy_t_20, accuracy_t_25]

    # precisions for top frequent
    precisions_frequent = [precision_base, precision_t_5, precision_t_10, precision_t_15, precision_t_20,
                           precision_t_25]

    # recalls for top frequent
    recalls_frequent = [recall_base, recall_t_5, recall_t_10, recall_t_15, recall_t_20, recall_t_25]

    # f1scores for top frequent
    f1scores_frequent = [f1score_base, f1score_t_5, f1score_t_10, f1score_t_15, f1score_t_20, f1score_t_25]

    # remaining words in the vocabulary for different frequencies.
    x_freq = [len(voc), len(voc_words_frequency_1), len(voc_words_frequency_5), len(voc_words_frequency_10),
              len(voc_words_frequency_15), len(voc_words_frequency_20)]
    # remaining words in the vocabulary for different percent of removed top frequent words.
    x_top = [len(voc), len(voc_words_frequent_5), len(voc_words_frequent_10), len(voc_words_frequent_15),
             len(voc_words_frequent_20), len(voc_words_frequent_25)]

    # plotting a graph for different frequencies.
    plt.figure()
    plt.subplot(1, 2, 1)
    x_temp = [0, 1, 2, 3, 4, 5]
    plt.xticks(ticks=x_temp, labels=x_freq, rotation=90)
    plt.plot(x_temp, accuracies_frequency, marker='D', markersize=13, label='accuracy')
    plt.plot(x_temp, precisions_frequency, marker='o', markersize=12, label='precision')
    plt.plot(x_temp, recalls_frequency, marker='s', markersize=11, label='recall')
    plt.plot(x_temp, f1scores_frequency, marker='^', markersize=10, label='f1score')
    plt.xlabel('remaining words in the vocabulary')
    plt.ylabel('performance')
    plt.legend()

    # plotting a graph for different percent of removed top frequent words.
    plt.subplot(1, 2, 2)
    plt.xticks(ticks=x_temp, labels=x_top, rotation=90)
    plt.plot(x_temp, accuracies_frequent, marker='D', markersize=13, label='accuracy')
    plt.plot(x_temp, precisions_frequent, marker='o', markersize=12, label='precision')
    plt.plot(x_temp, recalls_frequent, marker='s', markersize=11, label='recall')
    plt.plot(x_temp, f1scores_frequent, marker='^', markersize=10, label='f1score')
    plt.xlabel('remaining words in the vocabulary')
    plt.ylabel('performance')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # finishing message
    print("program has been finished.")
