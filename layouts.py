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
df2 = pd.read_csv('https://query.data.world/s/jq7lk27hbmlg2t5rf4tqoksxnrs4fl')

targets = df1['Emotion']
corpus = df1['Text']

corpus2 = df2['content']
targets2 = df2['sentiment']

list_emotions = list(df1['Emotion'].unique())
list_emotions.append('all')


#names = list('1er dataset', '2eme dataset', 'dataset global')
"""definition des graphs"""
#goBar emotions 1st data set
fig1 = go.Figure()
fig1 = go.Figure(data=[go.Histogram(x= targets, name='Emotions')], 
                     
               layout = {
                   'title':'Emotions Histogram',
                   'xaxis_title_text': 'Emotions',
                   'yaxis_title_text': 'frequence'})

#goBar emotions 2nd data set
fig2 = go.Figure()
fig2 = go.Figure(data=[go.Histogram(x= targets2, name='Emotions')], 
                     
               layout = {
                   'title':'Emotions Histogram',
                   'xaxis_title_text': 'Emotions',
                   'yaxis_title_text': 'frequence'})

#goBar emotions 2st data set
#fig3 = go.Figure()
#fig3 = go.Figure(data=[go.Histogram(x= targets2, name='sentiment')], 
#                     
#               layout = {
#                   'title':'Emotions Histogram',
#                   'xaxis_title_text': 'sentiment',
#                   'yaxis_title_text': 'frequence'})




"""layouts"""
Parag = dcc.Markdown('''Construit d’après les travaux du psychologue américain Robert Plutchik, 
             la roue des émotions est un modèle des émotions humaines et peut facilement 
             servir à définir des personnages, ainsi que leur évolution dans une trame 
             narrative. Est-il possible d identifier des émotions dans des phrases 
             narratives issues de communications écrites?''')
layout1 = html.Div(id="Nav_bar",children=[
    html.H1('La Roue des Emotions'),
    html.Nav(className = "nav-pills", children=[
        html.A('Home', className="nav-item nav-link btn", href='/'),
        html.A('Analyse des données brutes', className="nav-item nav-link btn", href='/1er_dataset'),
        html.A('Résultat des classifieurs', className="nav-item nav-link active btn", href='/2eme_dataset'),
        ]),
    html.H3('Contexte'),
    Parag,
    html.Img(style={'height':'60%','width':'50%','border-radius':'8px','padding':'50px'},
             id='image_roue',
             src=app.get_asset_url('roue.png')),
    ]),



"""2eme Page"""   
layout2= html.Div([
    html.Div(children=[
    html.Nav(className = "nav nav-pills", children=[
        html.A('Home', className="nav-item nav-link btn", href='/'),
        html.A('Analyse des données brutes', className="nav-item nav-link btn", href='/1er_dataset'),
        html.A('Résultat des classifieurs', className="nav-item nav-link active btn", href='/2eme_dataset'),
        ]),
    ]),
   
    html.Div([
          html.H2('Distribution des Emotions dans le Premier jeux de données'),
          dcc.Graph(id= 'graph_emotion', figure = fig1),
          
          html.H2('Histogramme des mots les plus fréquents du Premier jeux de données'),
          dcc.Graph(id = 'mots_hist'),
          dcc.RadioItems(
              id='radio_items',
              options = [{'label': k,'value': k} for k in list_emotions],
              value = "all",
              labelStyle={'display': 'inline-block'}),
                ]),
      #2nd Data Set
          html.Div([
          html.H2('Distribution des Emotions dans le Deuxième jeux de données'),
          dcc.Graph(id= 'graph_emotion2', figure = fig2),
          
          html.H2('Histogramme des mots les plus fréquents du Deuxième jeux de données'),
          dcc.Graph(id = 'mots_hist2'),
          dcc.RadioItems(
              id='radio_items2',
              options = [{'label': k,'value': k} for k in list_emotions],
              value = "all",
              labelStyle={'display': 'inline-block'}),
       
            dcc.Link('2eme Dataset', href='/2eme_dataset')
        
])

])
    

#3eme Page
layout3 = html.Div([
    html.Div(children=[
    html.Nav(className = "nav nav-pills", children=[
        html.A('Home', className="nav-item nav-link btn", href='/'),
        html.A('Analyse des données brutes', className="nav-item nav-link btn", href='/1er_dataset'),
        html.A('Résultat des classifieurs', className="nav-item nav-link active btn", href='/2eme_dataset'),
        ]),
    ]),

    html.H1('Analyse des performances'),


html.Div([
    dcc.Dropdown(id='choices',
    options=[
        {'label':'Data Set from Kaggle','value':'1st'},
        {'label':'Data Set from Data.World','value':'2nd'},
        {'label':'Global Data Set','value':'global'}
        ]),
              html.Hr(),
              html.Div(id='dd-output-container')]),

    html.H3(['Analyse des Performances des différents classifieurs']),
    html.P([
        html.Strong('1er data set:'),' Les résultats des 5 classifieurs donnent des score f1 compris entre 0,855 et 0.9, ces résultats efficaces sont expliqués essentiellement par un jeux de données propre. Au niveau des temps d éxécution, on remarque que l éxécution de la regression logistique est plus longue. A priori, cela est dû à son fonctionnement basé sur des classification binaires.'
        ]),
    html.P([    
        html.Strong('2eme data set:'),'Les Performances du deuxième jeux de données sont moins bonnes que celles du premier essentiellement à cause de la nature de ce dernier. En effet, le data set issue de data world est composé de vrais tweet, avec plusieurs classes (13 emotions différentes) la frontière entre les émotions est très fine ce qui rend la prédiction plus difficile pour un model.',
        ]),
    html.P([  
        html.Strong('Data set global:'),'On remarque que la concaténation de 2 jeux de données permet d améliorer les performances par rapport à celles du deuxieme jeux de données, cependant le temps d éxécution est nettement plus important à cause de la taille du fichié concaténé. On observe également que le classifier SGD fournis des meilleurs rélsultat en moins de temps que les autres models.'
       ])
        ])
    
        