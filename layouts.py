#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:08:17 2020

@author: wiem
"""
"""import librairies """
import dash_core_components as dcc
import dash_html_components as html
from app import app
import dash_table

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.decomposition import FastICA, KernelPCA, TruncatedSVD, SparsePCA, NMF, FactorAnalysis, LatentDirichletAllocation
#import nltk
import pickle

import plotly.graph_objs as go
#from wordcloud import WordCloud


""" import csv   """

df1 = pd.read_csv('data/emotion.csv')
targets = df1['Emotion']
corpus = df1['Text']

"""definition des graphs"""
#goBar emotions
fig1 = go.Figure()
fig1 = go.Figure(data=[go.Histogram(x= targets, name='Emotions')], 
                     
               layout = {
                   'title':'Emotions Histogram',
                   'xaxis_title_text': 'Emotions',
                   'yaxis_title_text': 'frequence'})

#goBar mots frequents        
#stopwords = nltk.corpus.stopwords.words("english")
vect = CountVectorizer(stop_words =english)
X = vect.fit_transform(corpus) 
words = vect.get_feature_names()
wsum = np.array(X.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels = [words[i] for i in ix]

def subsample(x, step=900):
    return np.hstack((x[:20], x[20::step]))
freq = subsample(wrank)
x = subsample(labels)
y = freq

fig2 = go.Figure()
fig2 = go.Figure(data= [go.Bar(x= subsample(labels),
                    y= freq,
                    name = "Words ordered by rank",
                    marker = dict(color = 'rgba(255, 174, 255, 0.5)'))],
                    layout = go.Layout(barmode = "group"))

"""layouts"""

layout1 = html.Div([
    html.H1('La Roue des Emotions'),
    html.Img(id='image_roue', src=app.get_asset_url('roue.png')),
    dcc.Graph(figure = fig1),
    dcc.Graph(figure = fig2),
    dcc.Link('2eme Dataset', href='/2eme_dataset')
])




""" 2eme Page"""
#layout2 = html.Div([
   # html.H3('Analyse des performances'),
    #dcc.Dropdown( id='Drop_analyse',
     #   options=[
      #      {'label': 'App 2 - {}'.format(i), 'value': i} for i in [
       #         'NYC', 'MTL', 'LA'
        #    ]]),
    
   # html.Div(id='app-2-display-value'),
    #dcc.Link('Go to Home Page', href='/')
#])
#"classifieur"
#targets1 = np.array([1 if x == "sadness" else 2 if x=="anger" else 3 if x=="love" else 4 if x=="surprise" else 5 if x=="fear" else 6 for x in targets1])



#layout2 = dash_table.DataTable(
#    id='table1',
 #    pipe11 = pickle.load(pipe1)
#)


layout3 = html.Div([
    html.H3('Home')
    
    ])
