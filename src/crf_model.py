import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *

# POS Tagging
from ner_pos_tagging import *

import sklearn_crfsuite
from sklearn.metrics import make_scorer, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn_crfsuite.utils import flatten
import scipy

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def sent2postag(sent):
    return [postag for token, postag, label in sent]

def get_features_and_classes(pos_lst):
    return [sent2features(s) for s in pos_lst],[sent2labels(s) for s in pos_lst]

def flat_f1_score(y_true, y_pred):
    return f1_score(flatten(y_true),flatten(y_pred),average='binary', pos_label='Y')

def flat_rc_score(y_true, y_pred):
    return recall_score(flatten(y_true),flatten(y_pred),average='binary', pos_label='Y')
    
def get_features_from_text(pos_tagged_text):
    return [sent2features(s) for s in pos_tagged_text]
     
def train_best_crm_model(X_train, y_train,save_model=False,cv=5,iter=50):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }
    
    f1_scorer = make_scorer(flat_f1_score)
     
    rs = RandomizedSearchCV(crf, params_space,
                        cv=5,
                        verbose=3,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
    
    rs.fit(X_train, y_train)
    
    crf = rs.best_estimator_
    
    if save_model:
        dump_pickle(crf,f'model/crf_model.pickle')
        
    printTS(f'best params  : {rs.best_params_}')
    printTS(f'best CV score: { rs.best_score_}')
    printTS(f'model size   : {rs.best_estimator_.size_ / 1000000:0.2f}M')

    return crf
    
def get_predicted_dishes(texts,results):
    dish_list = []
    dish_dict = {}
    
    sent_indexs = {}
    sent_texts  = {}
    
    k=0
    for i, sent in enumerate(texts):
        sent_indexs[i] = {}
        sent_texts[i]  = []    
        
        sent_index = {}
        sents = []    
        sent_part = []
        for j in range(len(sent)):
            sent_part.append(sent[j][0])
            sent_index[j]=k    
            if sent[j][0].strip() == '.':
                sents.append(' '.join(sent_part))
                sent_part = []
                k+=1
        
        sent_indexs[i] = sent_index
        sent_texts[i]  = sents      
        
    for i, sent in enumerate(texts):
        k = 0
        dish_dict[i] = {}
        dict_dish    = {}
        
        for j in range(len(sent)):
            
            if results[i][j] != 'Y' or j<=k:
                continue
            words = []
            while j < len(sent) and results[i][j] == 'Y':
                words.append(sent[j][0].title())
                j+=1
            k = j
            dish = ' '.join(words)
            dish_list.append(dish)
            dict_dish.setdefault(sent_texts[i][sent_indexs[i][j]],[]).append(dish)        

        dish_dict[i] = dict_dish
        #sent_val = ' '.join(sent_part)
        #dish_dict.setdefault(sent_val,[]).append(dish)
        #dish_dict[sent_val]= sent_list
                
    return dish_list, dish_dict
    
def find_dishes_with_crf(crf_model,str_arr):
    
    pos_taged_text = tag_pos_text(str_arr)
    features = get_features_from_text(pos_taged_text)
    results = crf_model.predict(features)
    return get_predicted_dishes(pos_taged_text,results)
    
        
if __name__ == "__main__":
    
    doc_dict = read_pickle('data/doc_dict.pickle')
    #new_dict = read_pickle('data/new_dict.pickle')
    #new_dict1 = read_pickle('data/new_dict1.pickle')
    
    #Dict_Merge(new_dict,doc_dict)
    #Dict_Merge(new_dict1,doc_dict)
    
    #pos_lst = tag_pos_from_dish_dict(doc_dict)
    #dump_pickle(pos_lst, f'model/tmp.pickle')
    pos_lst = read_pickle('model/tmp.pickle')
    
    #X_train, y_train = get_features_and_classes(pos_lst)
    #crf_model = train_best_crm_model(X_train, y_train,save_model=True,cv=3,iter=25)
    #dump_pickle(crf_model,f'model/crf_model.pickle')
        
    crf_model = read_pickle('model/crf_model.pickle')
    
    #test_str = 'Lots of meat Sausages, rosti (which is like a Circle of Hash-Brown-Potatoes), and big mugs of Beer. Is it really the swiss cheese? I would go in agaion for the tikka masala with baked beans'
    
    #texts = pre_process_data([test_str],lowercase_data=False)
    
    #pos_taged_text = tag_pos_text(texts)
    
    #features = get_features_from_text(pos_taged_text)
    #
    #results = crf_model.predict(features)
    
    #predictions, dish_dict = get_predicted_dishes(pos_taged_text,results)
    

    crf_sent_list, crf_dish_list = flat_dish_dict(doc_dict)
    reviews = load_review_data()[['categories']].copy()
    reviews['sents'] = crf_sent_list
    dump_pickle(reviews,'catdf.pickle')
    #dish_dict = get_ner_dishes(crf_sent_list,whitelist=predictions)
    
    #print(replace_best_matches(dish_dict,prefix='<== ',postfix=' ==>'))
#END