import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import re
import random
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def preprocessing_step():
    df = pd.read_csv(r'C:\Users\krishanu\Desktop\krishanupy\python\tf1.0\datasets\gossipcop_fake - Copy.csv')
    titles = [x for x in df['title'].values]
    tuple1 = processing_data(titles,'0')

    df = pd.read_csv(r'C:\Users\krishanu\Desktop\krishanupy\python\tf1.0\datasets\gossipcop_real - Copy.csv')
    titles2 = [x for x in df['title'].values]
    tuple2 = processing_data(titles2, '1')
    tuple1+=tuple2

    df = pd.read_csv(r'C:\Users\krishanu\Desktop\krishanupy\python\tf1.0\datasets\politifact_real - Copy.csv')
    titles2 = [x for x in df['title'].values]
    tuple2 = processing_data(titles2, '1')
    tuple1 += tuple2

    df = pd.read_csv(r'C:\Users\krishanu\Desktop\krishanupy\python\tf1.0\datasets\politifact_fake - Copy.csv')
    titles2 = [x for x in df['title'].values]
    tuple2 = processing_data(titles2, '0')
    tuple1 += tuple2

    df = pd.read_csv(r'C:\Users\krishanu\Desktop\krishanupy\python\tf1.0\datasets\train.csv')
    titles3 = [x for x in df['text'].values]
    values2 = [x for x in df['label'].values]

    tuple2 = processing_data2(titles2, values2)
    tuple1 += tuple2

    tuple1 = random.sample(tuple1, len(tuple1))
    #print(tuple1)
    print("hhll")
    return split_data(tuple1)

def processing_data2(titles,valus):
    processing_data2 = []
    for i in range(min(len(titles),len(valus))):
        processing_data2.append([titles[i], str(valus[i])])

    return processing_data2

def processing_data(titles,d_type):
    processing_data = []
    for single_data in titles:
        processing_data.append([single_data, d_type])

    return processing_data

def split_data(title):
    total = len(title)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []
    for indice in range(0,total):
        if indice< total*training_ratio:
            training_data.append(title[indice])
        else:
            evaluation_data.append(title[indice])

    return training_data, evaluation_data

def training_step(title, vectorizer):
    training_text = [data[0] for data in title]
    training_result = [data[1] for data in title]
    npa = np.asarray(training_text, dtype='U')
    training_text = vectorizer.fit_transform(npa)

    return BernoulliNB().fit(training_text, training_result)

def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))

def print_result(result):
    text, analyse_result = result
    print_text = "True" if analyse_result[0] == '1' else "False"
    print(text, ":", print_text)

def string_result(result):
    text, analyse_result = result
    print_text = "True" if analyse_result[0] == '1' else "False"
    return print_text

training_data, evaluation_data = preprocessing_step()
#print(training_data)
#vectorizer = TfidfVectorizer(binary = 'true')
#classifier = training_step(training_data,vectorizer)
#result = classifier.predict(vectorizer.transform(["donald trump"]))
#print_result( analyse_text(classifier,vectorizer,"Cindy with wig after dining"))
#print_result( analyse_text(classifier,vectorizer,"protests"))
#print_result( analyse_text(classifier,vectorizer,"Cindy"))
#print_result( analyse_text(classifier,vectorizer,"Cindy"))
#print_result( analyse_text(classifier,vectorizer,"Did Miley Cyrus and Liam Hemsworth secretly get married?"))
#print_result( analyse_text(classifier,vectorizer,"Teen Mom Star Jenelle Evans' Wedding Dress Is Available Here at $0"))