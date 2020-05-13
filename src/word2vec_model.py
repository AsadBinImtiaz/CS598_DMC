import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *

# Process_data
from process_data import *

# Import necessary libs
import pandas as pd
import gensim 
from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser

def get_wv_model_data(data):
    data = pre_process_data(data,lowercase_data=True)
    model_data = []  
    orig_data  = []
    skip = False
    x = 0
    for doc in nlp.pipe(iter(data), batch_size=1000, n_threads=20):
        for sent in doc.sents:
            sent_toks = []
            for i,token in enumerate(sent):
                if skip:
                    skip = False
                    continue
                    
                if token.is_stop:
                    continue
                        
                if (token.pos_ in w2v_model_pos):
                    sent_toks.append(token.lemma_.strip())
                                        
                if i+2<len(sent) and token.pos_ in ['NOUN','ADJ','PROPN'] and sent[i+1].pos_ in ['NOUN','PROPN'] and sent[i+2].pos_ in ['NOUN','PROPN']:
                    sent_toks.append(token.lemma_+"-"+sent[i+1].lemma_+"-"+sent[i+2].lemma_)
                    skip = True
                elif i>1 and i+1<len(sent) and token.pos_ in ['NOUN','PROPN'] and sent[i+1].pos_ in ['NOUN','PROPN'] and sent[i-1].pos_ not in ['PROPN','NOUN','ADJ']:
                    sent_toks.append(token.lemma_+"-"+sent[i+1].lemma_)
                    skip = True
                elif i+1<len(sent) and token.pos_ in ['ADJ'] and sent[i+1].pos_ in ['NOUN','PROPN']:
                    sent_toks.append(token.lemma_+"-"+sent[i+1].lemma_)
                    skip = True
            if len(sent_toks) != 0:
                model_data.append(sent_toks)                                
                orig_data.append(sent.text)
        x+=1
        if x%1000 == 0:
            printTS(f"Processed {x} texts")
    
    return model_data, orig_data

def find_wv_dish(data,model=None,blacklist=[]):
    
    if model == None:
        model = read_pickle('model/'+model_name+'.pickle')
    bl = [bk for bk in blacklist if len(bk.split()) == 1] + ['dish','food','cuisine']
    
    doc_dict = {}
    for i in range(len(data)):
        doc_dict[i] = {}    
        sents, osents = get_wv_model_data([data[i]])
        sent_dict = {}
        last_index = -9
        for j in range(len(sents)):
            for k in range(len(sents[j])):
                word = sents[j][k]
                try:
                    if (model.wv.similarity(w1='dish',w2=word)> 0.3 or model.wv.similarity(w1='cuisine',w2=word)> 0.3) and word not in bl:
                        sent_dict.setdefault(osents[j],[])
                        l = sent_dict[osents[j]]
                        if k == last_index + 1 and len(l) > 0 and l[-1].lower() + ' ' + word.lower() in osents[j].lower():
                            l[-1] = l[-1].title() + ' ' + word.title().replace('-',' ')
                        else: 
                            l.append(str(word).title().replace('-',' '))
                        sent_dict[osents[j]] = l
                        last_index = k
                except:
                    continue
            if osents[j] in sent_dict.keys():
                l = sorted(list(set(sent_dict[osents[j]])), key=len, reverse=False)
                for m in range(len(l)):
                    for n in range(len(l)-m-1):
                        if l[m] in l[n+m+1]:
                            l[m] = 'RemoveMe'
                           
                sent_dict[osents[j]] = sorted([o for o in l if o != 'RemoveMe' and o not in bl], key=len, reverse=True)
        if len(sent_dict)>0:
            doc_dict[i] = sent_dict
    return doc_dict

def create_w2v_model(sample_size=1000000,save_model=True):
    
    #reviews = load_review_data().iloc[:sample_size]
    #model_data, orig_data = get_wv_model_data(reviews['text'])    
    
    #dump_pickle(model_data,'model_data1.pickle')
    model_data = read_pickle('model_data1.pickle')
    
    printTS(f"Constructing W2V model")
    
    model = gensim.models.Word2Vec(
            model_data,
            size=1200,
            window=10,
            min_count=10,
            workers=100,
            iter=10)
    dump_pickle(model,'model/'+w2v_model_name+'.pickle')
        
    return model
    
if __name__ == "__main__":
    create_w2v_model(100000)
    
    dish_dict = find_wv_dish(['I order chicken fried rice or chicken tikka masala or dim sum. I did not like egg soup with butter masala and thai red curry. The server was good brown girl. table and chair meal'])
    print(replace_best_matches(dish_dict,prefix="</span><span style='color:#293795'><b>",postfix='</b></span><span>',use_orig=True))