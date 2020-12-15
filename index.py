#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:08:40 2020

@author: wiem
"""

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from layouts import layout1
import callbacks
from app import server

app.layout = html.Div(id ='app', children = [
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
         return layout1
    elif pathname == '/2eme_dataset':
         return layout2
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)