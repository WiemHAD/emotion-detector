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
from collections import defaultdict

import plotly.graph_objs as go
#from wordcloud import WordCloud


""" import csv   """

df1 = pd.read_csv('data/emotion.csv')
targets = df1['Emotion']
corpus = df1['Text']
list_emotions = list(df1['Emotion'].unique())
list_emotions.append('all')
#names = list('1er dataset', '2eme dataset', 'dataset global')
"""definition des graphs"""
#goBar emotions
fig1 = go.Figure()
fig1 = go.Figure(data=[go.Histogram(x= targets, name='Emotions')], 
                     
               layout = {
                   'title':'Emotions Histogram',
                   'xaxis_title_text': 'Emotions',
                   'yaxis_title_text': 'frequence'})
 

"""layouts"""
layout1 = html.Div(id="Nav_bar",children=[
     html.H1('La Roue des Emotions'),
    html.Nav(className = "nav-pills", children=[
        html.A('Go to Home', className="nav-item nav-link btn", href='/'),
        html.A('1st Data Set From Kaggle', className="nav-item nav-link btn", href='/1er_dataset'),
        html.A('2nd Data Set World Data', className="nav-item nav-link active btn", href='/2eme_dataset'),
        ]),
    
    html.Img(style={'height':'60%','width':'50%','border-radius':'8px','padding':'50px'},
             id='image_roue',
             src=app.get_asset_url('roue.png')),])
   
"""2eme Page"""   
layout2= html.Div([
    html.Div(children=[
    html.Nav(className = "nav nav-pills", children=[
        html.A('Go to Home', className="nav-item nav-link btn", href='/'),
        html.A('1st Data Set From Kaggle', className="nav-item nav-link btn", href='/1er_dataset'),
        html.A('2nd Data Set World Data', className="nav-item nav-link active btn", href='/2eme_dataset'),
        ]),
    ]),
   
      html.Div([
          html.H2('Histogramme des mots les plus fréquents'),
          dcc.Graph(id= 'graph_emotion', figure = fig1),
          dcc.Graph(id = 'mots_hist'),
          dcc.RadioItems(
              id='radio_items',
              options = [{'label': k,'value': k} for k in list_emotions],
              value = "all",
              labelStyle={'display': 'inline-block'}),
                ]),
       
            dcc.Link('2eme Dataset', href='/2eme_dataset'),
        
]) 
    

""" 3eme Page"""    
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

filename1='res1.joblib'
with open(filename1, 'rb') as f1:
        pickles1 = print_table1(pickle.load(f1))
    
filename2='res2.joblib'
with open(filename2, 'rb') as f1:
        pickles2 = print_table1(pickle.load(f1))
    
filename3='res3.joblib'
with open(filename3, 'rb') as f1:
        pickles3 = print_table1(pickle.load(f1))

     
layout3 = html.Div([
    html.Div(children=[
    html.Nav(className = "nav nav-pills", children=[
        html.A('Go to Home', className="nav-item nav-link btn", href='/'),
        html.A('1st Data Set From Kaggle', className="nav-item nav-link btn", href='/1er_dataset'),
        html.A('2nd Data Set World Data', className="nav-item nav-link active btn", href='/2eme_dataset'),
        ]),
    ]),

    html.H3('Analyse des performances'),
#commentaire#
 html.Div([
     dcc.Dropdown(
         id='choices',
         options=[{'label':'Data Set from Kaggle', 'value':'1st'},
                 {'label':'Data Set from Data.World', 'value':'2nd'},
                 {'label':'Global Data Set', 'value':'global'}],
        value='1st'),
     html.Hr(),
     html.Div(id='dd-output-container', children=[                
                  

    dash_table.DataTable(style_cell={'text-align': 'center','margin':'auto','width': '50px'},
        id='performances_premier_Dataset',
        columns=[{"name": i, "id": i} for i in pickles1.columns],
        data=pickles1.to_dict('records'), 
       editable=True),
    
    
     dash_table.DataTable(style_cell={'text-align': 'center','margin':'auto','width': '50px'},
        id='performances_deuxieme_Dataset',
        columns=[{"name": i, "id": i} for i in pickles2.columns],
        data=pickles2.to_dict('records'), 
       editable=True),
     
     
      dash_table.DataTable(style_cell={'text-align': 'center','margin':'auto','width': '50px'},
        id='performances_global_Dataset',
        columns=[{"name": i, "id": i} for i in pickles3.columns],
        data=pickles3.to_dict('records'), 
       editable=True),
    #style_cell={"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
])]),   
    
   html.Div([
        dcc.Link('Go to Home Page', href='/')]),
    
    html.Div(id="app-2-display-value")  
    
    ])
