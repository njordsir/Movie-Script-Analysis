# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:55:27 2016

@author: Siddharth Jar jarsiddharth@iitkgp.ac.in

The script extracts sentiment/emotion from movie script in .txt format and 
provides sentiment/emotion quotient for the movie.

INPUT:
dirpath:    Directory where movie scripts are stored. (.txt format)
nrc_lex:    NRC lexicon for sentiment/emotion keywords (.csv format)
            Link to webpage: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
N:          Number of bins in which movie script is to be divided
OUTPUT:
emotion_dict:   Dictionary with movie names as keys and doctionary of 
                sentiment/emotion quotient as values

The code first breaks script into N bins of equal number of lines. These bins 
are then processed for sentiment/emotion quotient based on keywords

Emotion quotient is obtained by normalizing word count by total word count of 
emotion words. Similarly for sentiment word count is normalized by total word 
count of sentiment keywords.

NB: COde fragment for pickling as has been commented. It can be used as desired
"""


from os import listdir
from os.path import isfile, join
from fileinput import input
import pandas as pd
import pickle
import re


nrc_lex = pd.read_csv('~/nrc_lexicon.csv')
dirPath = '~\imsdb/'
N = 200

all_script_bins = {}
textFiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

for script in textFiles:
    print script[:-4]
    if('.' in script[:-4]):
        continue
    f = open(dirPath+script, 'r')
    text = f.read()
    lines = text.splitlines()
    noLines = len(lines)
    noPara = noLines/N
    lines = [re.sub(r'<.*?>','',l.lower()) for l in lines]
    script_bins = [""]*N
    for i in range(1,N+1):
        if(i==1): 
            text = ' '.join(lines[0:i*noPara - 1])
        else:
            text = ' '.join(lines[(i-1)*noPara:i*noPara - 1])
        script_bins[i-1]= " ".join(text.split())    
        
    all_script_bins[script[:-4]] = script_bins
 
  
#with open('~\imsdb_bins\script_binwise.pkl','w') as fp: 
#    pickle.dump(all_script_bins,fp)
#  


emotions = list(nrc_lex.category.unique())
emotion_dict = {}

#with open('~\imsdb_bins\script_binwise.pkl' , 'r') as fp:
#    all_script_bins = pickle.load(fp)

for script in all_script_bins:
    print script
    text = all_script_bins[script]
    emotion_dict[script] = {}
       
    for e in emotions:
        emotion_dict[script][e] = 0.0
    for t in text:     
        for e in emotions:
                emotion_dict[script][e] += len(set(nrc_lex[(nrc_lex.category == e) & (nrc_lex.score == 1)]['word'].tolist()) & set(t.split()))
                
    total_e = 0.0
    total_s = 0.0
    for e in emotions:
        if ((e == "positive") | (e == "negative")):
            total_s += emotion_dict[script][e]
        else:
            total_e += emotion_dict[script][e]
    if ((total_s ==0) | (total_e == 0)):
        continue
    for e in emotions:
        if ((e == "positive") | (e == "negative")):
            emotion_dict[script][e] = round(emotion_dict[script][e]/total_s,3)
        else:
            emotion_dict[script][e] = round(emotion_dict[script][e]/total_e,3)    
            
            
#with open('~\imsdb_bins\emotion_binwise.pkl','w') as fp: 
#    pickle.dump(emotion_dict,fp)          

