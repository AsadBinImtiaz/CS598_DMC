#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dir_home    = ''
dir_start   = 'src/'
dir_data    = 'data/'
dir_model   = 'model/'

spacy_lib      = 'en_core_web_sm'

negations      = ['not','barely','rarely','no','hardly','never','seldom']
               
select_labels  = ['PERSON','ORG','PRODUCT','WORK_OF_ART']
connect_words  = ['and', 'with', '&', 'n']
               
non_starters   = ('1','2','3','4','5','6','7','8','9','0','about ', 'am')
               
non_enders     = (' restaurant', ' bar', ' hotel', ' pub', ' shop', ' mall', ' cafe', ' establishment', ' company',' service',' atmosphere', 
                  ' menu', ' chains', ' options',' experience', ' group', ' staff', ' club', ' hour', ' lunch', ' dinner', 
                  ' breakfast', ' stars', 'place', ' 1', ' 2', ' 3' , ' 4', ' 5', ' 6', ' 7', ' 8', ' 9' ' 0', ' yelp')
               
start_trimmers = ('a ','& ', 'an ', 'the ','good ','best ','great ','awesome ','bad ','worst ', 'excellent ', 'i ', 
                  'extreme ','amazing ',' awful', 'better ', 'famous ', 'fantastic ','favorite ','giant ','tasty ')
               
end_trimmers   = (' &',' dollar', ' dollars', ' i', ' a', ' an', ' or')

fil_symbols    = ['/','*','(',')','|','_']

connect_words  = ['and', 'with', '&', 'n']

match_ratio    = 90

sentiment_pos  = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'DET', 'ADP', 'CCONJ', 'INTJ']

w2v_model_pos  = ['PROPN', 'NOUN', 'ADJ' ]

w2v_model_name = 'w2v_model'



