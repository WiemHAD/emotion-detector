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

def subsample(x, step=900):
    return np.hstack((x[:20], x[20::step]))

@app.callback(
    Output('app-1-display-value', 'children'),
    Input('image_roue', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)



@app.callback(
    Output('mots_hist', 'figure'),
    Input('radio_items', 'value'))
def make_mots_hist(value):
    if value == 'all':
        df0 = df1
    else:
        df0 = df1.loc[df1.Emotion == value]

    vect = CountVectorizer(stop_words = 'english')
    X = vect.fit_transform(corpus) 
    words = vect.get_feature_names()
    wsum = np.array(X.sum(0))[0]
    ix = wsum.argsort()[::-1]
    wrank = wsum[ix] 
    labels = [words[i] for i in ix]
    vect = CountVectorizer(stop_words = 'english')
    X = vect.fit_transform(df0.Text) 
    words = vect.get_feature_names()
    wsum = np.array(X.sum(0))[0]
    ix = wsum.argsort()[::-1]
    wrank = wsum[ix] 
    labels = [words[i] for i in ix]

    trace = go.Bar(x = subsample(labels), 
                   y = subsample(wrank),
                   marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                   line = dict(color ='rgb(0,0,0)',width =1.5)))
    layout = go.Layout(
                    xaxis_title_text = 'Word rank',
                    yaxis_title_text = 'word frequency')
                  
    figure = go.Figure(data = trace, layout = layout)
    return figure

@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('choices','value')]
    )
def update_date_dropdown(choices):
        return [{'label':'Data Set from Kaggle', 'value':'1st'},
                 {'label':'Data Set from Data.World', 'value':'2nd'},
                 {'label':'Global Data Set', 'value':'global'}]
    
def set_display_children(selected_value):
    if selected_value =='1st':
        return 'performances_premier_Dataset'
    elif selected_value =='2nd':
        return 'performances_deuxieme_Dataset'
    elif selected_value =='global':
        return 'performances_global_Dataset'
    else:
        return 'you have selected {} option'.format(selected_value)



@app.callback(
    Output('app-2-display-value', 'children'),
    Input('Les performances du premier Dataset', 'value'))
def display_value1(value):
   return 'You have selected "{}"'.format(value)