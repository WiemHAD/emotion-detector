#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:07:16 2020

@author: wiem
"""

import dash
import dash_html_components as html

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
html.Img(src=app.get_asset_url('roue.png'))