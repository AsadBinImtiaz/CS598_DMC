#!/usr/bin/env python
# coding: utf-8

import os
import sys

# Util Functions
from util_funcs import *
    

def standardize_dish(dish):
    dish = dish.replace('"','').replace("'",'').replace(".",'').replace("-",' ').strip()
    dl   = dish.lower()
    
    if dl.startswith(non_starters):
        dish = ''
    if dl.startswith(start_trimmers):
        dish = ' '.join(dish.lower().split()[1:])
    if dl.endswith(end_trimmers):
        dish = ' '.join(dish.lower().split()[:-1])
    if dl.endswith(non_enders):
        dish = ''
    
    dish = dish.title().replace('  ',' ').replace('  ',' ')
    #dish = ' '.join(list_diff(dish.lower().split(),['the'])).title()
    chk  = ['/','*','(',')','|','_']
    if any(ch in dish for ch in fil_symbols) or len(dish) < 3:
        dish = ''
    
    return dish      
    
def trim_names_and_ampersand(name):   
    name = name.replace('&','and')    
    if name.startswith(('-','=',"'")):
        name = name[1:]
    #if len(name.split())> 1 and name.split()[-1] == name.split()[-2]:
    #    name = ' '.join(name.split()[:-1]
    return name                       
           
def pre_process_data(data,remove_symbols=True,remove_newlines=True,lowercase_data=False):
    printTS (f"Pre-Processing data: rows={len(data)}")
    
    printTS (f"Removing URLs")
    data = remove_urls(data)
    
    printTS (f"Converting short nots to long")
    data = remove_short_nots(data)
    
    if remove_symbols:
        printTS (f"Removing Symbols")
        data = remove_symbs(data)
    
    if remove_newlines:
        printTS (f"Removing Newlines")
        data = remove_linebreaks(data)
    
    if lowercase_data:
        printTS (f"Lowercase Sentences")
        data = lowercase_list (data)
    
    printTS (f"Removing Extra Spaces")
    data = remove_spaces(data)
    
    return data
    
def process_select_list():
    reviews = load_review_data()
     
    select_dict = {}
    for index, row in reviews.iloc[:].iterrows():
        state = row['state']
        city  = row['city']
        name  = row['name']
        revi  = row['review_id']
        text  = row['text']
    
        select_dict.setdefault(state, {})    
        city_dict = select_dict[state]
        
        city_dict.setdefault(city, {})    
        name_dict = city_dict[city]
    
        name_dict.setdefault(name, []).append(revi)    
        
    dump_pickle(select_dict,f'data/select_list.pickle')
    
def flat_dish_dict(dish_dict):
    sent_list = []
    dish_list = []
    for text_id, sent_dict in dish_dict.items():

        text_sents  = ''
        text_dishes = []
        for sent,dish_lst in sent_dict.items():
            text_sents = text_sents +' '+ sent
            for dish in dish_lst:
                text_dishes.append(dish)
        sent_list.append(text_sents)
        dish_list.append(text_dishes)
        
    return sent_list, dish_list
    
if __name__ == "__main__":
    process_select_list()
