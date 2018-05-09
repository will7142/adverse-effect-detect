#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 08 20:13:14 2018

@author: will7142 (Will Rosenfeld)

Purpose: To pull data from several FDA open data resources:
    1. FAERS: https://open.fda.gov/drug/event/reference/
    2. VAERS: https://wonder.cdc.gov/vaers.html
    
API Key: nf2NArROcmgqc4N3OZPHSiZYKVTSIDWhygRuNXT7
"""

# imports
import requests
from urllib import requests, urlopen
import pandas as pd
import json
from pandas.io.json import json_normalize

# test AE table creation
data = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.pharm_class_epc:"nonsteroidal+anti-inflammatory+drug"&count=patient.reaction.reactionmeddrapt.exact')
data.json()
rslt = data.json()['results']
rslt

# convert to DataFrame
df = pd.io.json.json_normalize(rslt)
df

# test AE table creation
data = requests.get('https://api.fda.gov/drug/event.json?search=patient.drug.openfda.brand_name:"Humira"&count=patient.reaction.reactionmeddrapt.exact')
data.json()
rslt = data.json()['results']
rslt

# convert to DataFrame
df = pd.io.json.json_normalize(rslt)
df