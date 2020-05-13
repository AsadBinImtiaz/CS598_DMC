#!/usr/bin/env python
# coding: utf-8

# Import necessary libs
import pandas as pd
import pickle as pickle
import re
import itertools
import warnings
from datetime import datetime


# Spacy
import spacy
from spacy.lang.en import English

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# logging
import logging

# Config
from app_config import *

#
# ## Helpful Functions
# 

logEnabled = 0

print('Loading Spacy NLP large English library: '+spacy_lib, end =" ")
nlp   = spacy.load(spacy_lib)
nlp2  = spacy.load(spacy_lib)
print('[Done]')

nlp.vocab["not"].is_stop = False

stopwords = spacy.lang.en.stop_words.STOP_WORDS
stopwords.remove('not')
stopwords.add('ll')
stopwords.add('ve')

# Print message with Timestamp
def printTS(strInput):
    print (str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ': ' + str(strInput))
    logOutput(strInput)
    
def logOutput(strInput,newLineRep=""):
    #if logEnabled == 1:
    logging.info(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ': ' + str(strInput).replace("\n",newLineRep))

log_file  = dir_home+"logs/"+str(datetime.now().strftime("%Y%m%d%H%M%S"))+".log"

def read_pickle(filename):
    pk = None
    printTS('loading: '+dir_home+filename)
    try:
        with open(dir_home+filename, 'rb') as f:
            pk = pickle.load(f)
    except:
        printTS('Error loading: '+dir_home+filename)
        pass
    return pk
    
def dump_pickle(obj,filename):
    printTS('saving: '+dir_home+filename)
    try:
        with open(dir_home+filename, 'wb') as f:
            pickle.dump(obj,f)
    except:
        printTS('Error saving: '+dir_home+filename)
        pass

#
# Save Dafaframe to CSV on S3 project bucket
#
def df_to_csv(df,filename,encoding='utf-8'):
    df.to_csv(dir_home+filename,encoding=encoding)

#
# Read Dafaframe from CSV on S3 project bucket
#
def csv_to_df(filename,encoding='utf-8'):
    return pd.read_csv(dir_home+filename,encoding=encoding).drop(labels='Unnamed: 0', axis=1)

#
# Get difference between 2 lists
#
def list_diff(list1,list2):
    return list(itertools.filterfalse(set(list2).__contains__, list1)) 

#
# Remove Stop-Words from list of data
#    
def remove_stop_words(data):
    sents = [str(sent).lower().split() for sent in data]
    sents = [' '.join(d) for d in [list_diff(sent,stopwords) for sent in sents] ]
    for i,rev in enumerate(data):
        if len(sents[i].split())<=5:
            sents[i] = rev
    return sents
            
#
# Remove URLs from list of data
#    
def remove_urls (data):
    repeat_regexp = re.compile(r'(\w)(\1{2,})', re.IGNORECASE)
    repl = r'\1'
    data = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', str(sent) , flags=re.MULTILINE | re.IGNORECASE) for sent in data]
    data = [re.sub(repeat_regexp, repl, str(sent)) for sent in data]
    data = [re.sub(r'(\d+/\d+/\d+)',' ', str(sent)) for sent in data]
    return(data)
 
# 
##### Tokenization
# 
def tokenize_docs(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True re

#
# Remove new lies and symbols & lowercase from list of data
#
def remove_linebreaks(data):
    data = [str(sent).replace('\\n','.').replace('\n','.').replace('..','.').replace('..','.') for sent in data]
    data = [str(sent).replace('\\r','.').replace('\r','.').replace('..','.').replace('..','.') for sent in data]
    data = [str(sent).replace('..','.').replace('.',' . ').replace('  ',' ').replace('  ',' ') for sent in data]
    data = [str(sent+'.').replace('  ',' ').replace('. .','.').replace('..','.').replace('.',' . ') for sent in data]
    return data

#
# Remove punctuation list of data
#
def remove_puncts(data):
    data = [str(sent).replace('\\n',' ').replace('\n',' ').replace(',',' ').replace('?','.').replace('!','.').replace("\\'","''") for sent in data]
    data = [str(sent).replace(';','.').replace('\r',' ').replace(':','.').replace('/',' ').replace('"','').replace('$','dollars ') for sent in data]
    data = [str(sent).replace('~','').replace('(',' ').replace(')',' ').replace('+',' ').replace('#',' ').replace('-',' ').replace('%',' ') for sent in data]
    data = [str(sent).strip('*').strip('-').replace('=',' ').replace('@',' ').replace('^',' ').replace('  ',' ').replace('  ',' ') for sent in data]
    return data
   
#
# Remove spaces and symbols from list of data
#
def remove_symbs(data):
    data = [str(sent).replace(';','.').replace('`',' ').replace('?','.').replace('!','.') for sent in data]
    data = [str(sent).replace(':','.').replace('@','').replace('^','').replace('$','dollars ') for sent in data]
    data = [str(sent).replace('~','').replace('+',' ').replace('#',' ').replace('%',' ') for sent in data]
    data = [str(sent).strip('*').strip('-').replace('=',' ').replace('  ',' ').replace('  ',' ') for sent in data]
    data = [d.replace('"','').replace("'",'').replace("-",' ').replace("  ",' ').replace("  ",' ').strip() for d in data]
    return data

#
# Remove spaces and symbols from list of data
#
def remove_spaces (data):
    data = [re.sub(r'\s+', ' '  ,  str(sent)) for sent in data]
    return data

#
# Convert n't to not in list of data
#
def remove_short_nots (data):
    data = [re.sub("won't", 'would not', str(sent)) for sent in data]
    data = [re.sub("can't", 'can not', str(sent)) for sent in data]
    data = [re.sub("n't", ' not', str(sent)) for sent in data]
    data = [re.sub("i'm", 'i am', str(sent)) for sent in data]
    data = [re.sub("'ll", ' will', str(sent)) for sent in data]
    data = [re.sub("'ve", ' have', str(sent)) for sent in data]
    data = [re.sub("'s" , 's'   , str(sent)) for sent in data]
    return data
 
#
# Split list of sentences on space
#
def split_on_space (data):
    data = [sent.split() for sent in data]
    #data = list(tokenize_docs(data))
    return data
            
#
# Lowercase list of sentences
#
def lowercase_list (data):
    return [sent.lower() for sent in data]    

def remove_stop_words_fast(data):
    start = time.time()
    stops = get_stop_word_list()
    printTS(f"Stopwords removed - took {time.time() - start:9.6f} secs")
    return [list_diff(sent,stops) for sent in data]

def find_sub_list(sub_list,this_list):
    tl = [s.lower() for s in this_list]
    sl = [s.lower() for s in sub_list]
    try:
        return (tl.index(sl[0]),len(sl)) 
    except:
        return (-1,-1)

def start_logger():
    warnings.filterwarnings("ignore")
    try:
        Path(log_file).touch()   
        logging.basicConfig(filename=log_file,level=logging.INFO)
        logEnabled = 1
        printTS ("------------------START--------------------")            
    except Exception as SomeError:
        printTS (f"Warning: logger initialization falied: {str(SomeError)}")
    

def load_dish_blacklist():
    return read_pickle(dir_data+'blacklist.pickle')
        
def load_dish_whitelist():
    return read_pickle(dir_data+'whitelist.pickle')
        
def load_top_500_topmine_dishes():
    return read_pickle(dir_data+'topmine_top_500_dishes.pickle')

def Dict_Merge(dict1, dict2): 
    return(dict2.update(dict1)) 
    
def get_best_match(str1,str2):
    if fuzz.ratio(str1.lower(),str2.lower()) >= match_ratio:
        return str1
    return None
        
def replace_best_match(out_str,in_str,prefix='',postfix='',use_orig=False):
    l = len(in_str.split()) # Length to read orig_str chunk by chunk
    splitted = out_str.split()
    
    m = len(splitted)-l+1
    if m>0:
        splitted[0] = splitted[0].title()    
    to_add = 0
    if prefix != '':
        to_add += len(prefix.split())-1
    if postfix != '':
        to_add += len(postfix.split())-1
    
    i = 0
    while i < m:
        test = " ".join(splitted[i:i+l])
        if fuzz.ratio(in_str.lower(), test.lower().replace('& ','and ').replace('..','. .')) >= match_ratio :
            before = " ".join(splitted[:i])
            after = " ".join(splitted[i+l:])
            
            if use_orig:
                out_str = before+" "+prefix+test.title()+postfix+" "+after
            else:
                out_str = before+" "+prefix+in_str.title()+postfix+" "+after    
                
            splitted = out_str.split()
            m = len(splitted)-l+1
            i+=(to_add)    
        i+=1
        
    return out_str
    
def replace_best_matches_in_text(sent,dish_list,prefix='',postfix='',use_orig=False):
    out_str = sent
    in_str_ar = sorted(list(set(dish_list)), key=len, reverse=True)

    for in_str in in_str_ar:
        out_str = replace_best_match(out_str,in_str,prefix=prefix,postfix=postfix,use_orig=use_orig)
            
    return out_str    
    

def replace_best_matches(out_dicts,prefix='',postfix='',use_orig=False):
    new_dicts = {}
    for text_id,out_dict in out_dicts.items():
        new_dict = {}
        for sent,dish_list in out_dict.items():
            out_str = sent
            in_str_ar = sorted(list(set(dish_list)), key=len, reverse=True)
            for in_str in in_str_ar:
                out_str = replace_best_match(out_str,in_str,prefix=prefix,postfix=postfix,use_orig=False)
            new_dict[out_str] = dish_list
        
        new_dicts[text_id] = new_dict
    return new_dicts

    
def load_review_data():
    return read_pickle(f'data/review_data.pickle')

global_BL      = load_dish_blacklist()
global_WL      = load_top_500_topmine_dishes()
        
if __name__ == "__main__":
    start_logger()
    printTS ("Initializated")
       
#END

