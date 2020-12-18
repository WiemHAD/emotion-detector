#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:07:49 2020

@author: wiem
"""

from dash.dependencies import Input, Output
import dash
from app import app
from layouts import layout1, layout2

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
from collections import defaultdict

import plotly.graph_objs as go

df1 = pd.read_csv('data/emotion.csv')
corpus = df1['Text']

df2 = pd.read_csv('https://query.data.world/s/jq7lk27hbmlg2t5rf4tqoksxnrs4fl')
df2= df2.drop(['author','tweet_id'], axis=1)
corpus2 = df2['content']
targets2 = df2['sentiment']

def subsample(x, step=900):
    return np.hstack((x[:20], x[20::step]))

def print_table1(res1):
    # Compute mean 
    final = {}
    for model in res1:
        arr = np.array(res1[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            "Precision" : arr[:,2].mean().round(3),
            "Recall" : arr[:,2].mean().round(3)
        }
    df4 = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df4

@app.callback(
    Output('app-1-display-value', 'children'),
    Input('image_roue', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)


#histogramme mots frequents 1st dataset
@app.callback(
    Output('mots_hist', 'figure'),
    Input('radio_items', 'value'))
def make_mots_hist(value):
    if value == 'all':
        dfk = df1
    else:
        dfk = df1.loc[df1.Emotion == value]
        
    vect = CountVectorizer(stop_words = 'english')
    X = vect.fit_transform(dfk.Text) 
    words = vect.get_feature_names()
    wsum = np.array(X.sum(0))[0]
    ix = wsum.argsort()[::-1]
    wrank = wsum[ix] 
    labels = [words[i] for i in ix]

    trace1 = go.Bar(x = subsample(labels), 
                   y = subsample(wrank),
                   marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                   line = dict(color ='rgb(0,0,0)',width =1.5)))
    layout = go.Layout(
                    xaxis_title_text = 'Word rank',
                    yaxis_title_text = 'word frequency')
                  
    figure = go.Figure(data = trace1, layout = layout)
    return figure




@app.callback(
    Output('mots_hist2', 'figure'),
    Input('radio_items2', 'value'))
def make_mots_hist2(value):
    if value == 'all':
        dfw = df2
    else:
        dfw = df2.loc[df2['sentiment'] == value]
        
    vect = CountVectorizer(stop_words = 'english')
    X = vect.fit_transform(dfw['content']) 
    words = vect.get_feature_names()
    wsum = np.array(X.sum(0))[0]
    ix = wsum.argsort()[::-1]
    wrank = wsum[ix] 
    labels = [words[i] for i in ix]

    trace2 = go.Bar(x = subsample(labels), 
                   y = subsample(wrank),
                   marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                   line = dict(color ='rgb(0,0,0)',width =1.5)))
    layout = go.Layout(
                    xaxis_title_text = 'Word rank',
                    yaxis_title_text = 'word frequency')
                  
    figure = go.Figure(data = trace2, layout = layout)
    return figure

@app.callback(
     Output('dd-output-container','children'),
     Input('choices','value'))
def set_display_children(selected_value):
    if selected_value =='1st':
        filename1='res1.joblib'
        with open(filename1, 'rb') as f1:
            pickles1 = print_table1(pickle.load(f1))
            table1=dash_table.DataTable(style_cell={'text-align': 'center','margin':'auto','width': '50px'},
                                       id='performances_premier_Dataset',
                                       columns=[{"name": i, "id": i} for i in pickles1.columns],
            data=pickles1.to_dict('records'), 
            editable=True), 
            
            return table1

    elif selected_value =='2nd':
        filename2='res2.joblib'
        with open(filename2, 'rb') as f1:
            pickles2 = print_table1(pickle.load(f1))
            table2=dash_table.DataTable(style_cell={'text-align': 'center','margin':'auto','width': '50px'},
                                 id='performances_deuxieme_Dataset',
                                 columns=[{"name": i, "id": i} for i in pickles2.columns],
                                 data=pickles2.to_dict('records'), 
                                 editable=True),
            return table2
    else: 
        filename3='res3.joblib'
        with open(filename3, 'rb') as f1:
            pickles3 = print_table1(pickle.load(f1))
            table3 = dash_table.DataTable(style_cell={'text-align': 'center','margin':'auto','width': '50px'},
                                          id='performances_global_Dataset',
                                          columns=[{"name": i, "id": i} for i in pickles3.columns],
                                          data=pickles3.to_dict('records'), 
                                          editable=True),
            return table3

@app.callback(
    Output('app-2-display-value', 'children'),
    Input('Les performances du premier Dataset', 'value'))
def display_value1(value):
   return 'You have selected "{}"'.format(value)