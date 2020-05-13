import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *

# Process_data
from process_data import *

reviews = None

def prep_text(data):
    docs = []
    
    i = 0
    xx = int(len(data)/10)
    if xx==0:
        xx = 1;
        
    for doc in nlp.pipe(iter(data), batch_size=1000, n_threads=20):
        pdoc = []
        for sent in doc.sents:
            cur = 0
            tot = len(sent)-1
            for token in sent:
                if cur < tot - 1 and sent[cur].pos_ == 'ADJ' and sent[cur+1].pos_ in ['NOUN','PROPN']:
                    pdoc.append(token.text.title())
                #elif cur-1 > 0 and sent[cur].pos_ == 'NOUN' and sent[cur-1].pos_ in ['ADJ']:
                #    pdoc.append(token.text.title())
                elif token.pos_ in ['PROPN',]:
                    pdoc.append(token.text.title())
                else:
                    pdoc.append(token.text.lower())
                cur+=1
        docs.append(' '.join(pdoc))
        
        i+=1
        
        if i% xx == 0:
            printTS(f"Cleansed: {i} review(s)")
            
    return docs
    
def find_dishes_with_ner(data,name=[],city=[],blacklist=[],whitelist=[],startnum = 0):
    
    new_data = []
    
    i = startnum
    xx = len(data[i:])
    printTS(f"Total text(s): {xx}")

    doc_dict = {}
    
    name = lowercase_list(name)
    city = lowercase_list(city)
    
    for doc in nlp.pipe(iter(data[startnum:]), batch_size=1000, n_threads=20):
        
        sentns = {}
        dishes = []
        doc_dict[i] = sentns
        
        for sent in doc.sents:
            sent_dishes = []

            if len(sent) <=2:
                continue
            x = 0
            for ent in sent.ents:
                
                if (x > ent.start_char) or (len(name) == len(data) and ent.text.lower() in name[i]) or (ent.text.lower() in name) or (ent.text.lower() in city):
                    continue
                if ent.label_ in select_labels and len(ent.text) > 3:
                    #print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    sent_start = len(str(doc)[ent.start_char:ent.end_char].split())
                    
                    doc2 = nlp2(str(doc)[ent.start_char:sent.end_char])
                    sent_end = len(str(doc2).split())
                    
                    add = []
                    while sent_start<sent_end:
                        if doc2[sent_start].pos_ in ['NOUN','PROPN']:
                            if ent.text.split()[-1].lower() != doc2[sent_start].text.lower():
                                add.append(str(doc2[sent_start]))
                        elif sent_start+1<sent_end and doc2[sent_start].text.lower() in connect_words and doc2[sent_start+1].pos_ in ['NOUN','PROPN']: 

                            add.append(str(doc2[sent_start]))
                            add.append(str(doc2[sent_start+1])) 
                            sent_start+=1
                        else:
                            break
                        sent_start+=1

                    dish = standardize_dish(standardize_dish(str(ent)+" "+" ".join(add)))
                    if len(dish) < 3:
                        break
                    
                    doc2 = nlp2(dish)
                    if len(doc2.ents) == 0 or (doc2.ents[0].label_ in select_labels and doc2.ents[-1].label_ in select_labels):
                        dish = trim_names_and_ampersand(dish)
                        dsen = trim_names_and_ampersand(sent.text)
                        if dish.lower() not in blacklist:
                            x=len(dish)+ent.start_char
                            doc_dict[i].setdefault(dsen, []).append(dish)
                            sent_dishes.append(dish.lower())

            if whitelist:
                for x in whitelist:
                    if ' '+x.lower()+' ' in ' '+sent.text.replace('&','and').lower()+' ' and len([s for s in sent_dishes if any(xs in s for xs in [x.lower()])]) == 0:
                        dish = x.title()
                        dsen = trim_names_and_ampersand(sent.text)
                        
                        doc_dict[i].setdefault(dsen, []).append(dish)
                        sent_dishes.append(x.lower())
                    if not x.endswith('es'):
                        x = x+'s'
                        if ' '+x.lower()+' ' in ' '+sent.text.replace('&','and').lower()+' ' and len([s for s in sent_dishes if any(xs in s for xs in [x.lower()])]) == 0:
                            dish = x.title()
                            dsen = trim_names_and_ampersand(sent.text)
                        
                            doc_dict[i].setdefault(dsen, [])
                            if dish not in doc_dict[i][dsen]:
                                doc_dict[i][dsen].append(dish)
                            sent_dishes.append(x.lower())
                    

        i+=1
        
        if i%1000 == 0:
            printTS(f"Processed: {i} / {xx}")
                                
    return doc_dict

def process_review_data(df_reviews,startnum=0,sample_size=1000000):
    reviews = load_review_data().iloc[:sample_size]
        
    # Review text
    printTS(f'Preprocessing {len(reviews[startnum:])} reviews')
    data = pre_process_data(df_reviews['text'][startnum:])
    
    # Restaurant Name list
    printTS(f'Preprocessing {len(reviews[startnum:])} names')
    name = pre_process_data(df_reviews['name'][startnum:],remove_symbols=False)
    
    # Lowercase city name list
    printTS(f'Preprocessing {len(reviews[startnum:])} cities')
    city = lowercase_list(df_reviews['name'][startnum:])
    
    blacklist = load_dish_blacklist()
    #whitelist = load_top_500_topmine_dishes()
    whitelist = load_dish_whitelist()
    
    return get_ner_dishes(data,name,city,blacklist=blacklist,whitelist=whitelist)

def tag_pos(sent):
    doc = nlp(sent)
    tok = []
    tag = []
    for token in doc:
        tok.append(token.text)   
        tag.append(token.tag_)
    return tok,tag

def tag_pos_text(arr_sents):
    sent_list = []
    for sent in arr_sents:
        tokens, postags = tag_pos(sent)
        sent_list.append(list(zip(tokens, postags)))
    return sent_list

def tag_pos_from_dish_dict(doc_dict):
    
    pos_lst = []
    x = 0
    
    whitelist = load_dish_whitelist() ####
    whitelist = whitelist + [wl+'s' for wl in whitelist if not wl.endswith('es')]
    
    for text,sents in doc_dict.items():
        if len(doc_dict[text]) > 0:
            sent_dict = doc_dict[text]
            
            for sent,dish in sent_dict.items():
                
                dish = [d.strip().replace('-',' ') for d in dish]
                sent = sent.replace('dont','do not')
                sent_val = str(' '+sent+' ').replace(' nt ',' not ').replace(' ll ',' will ').replace(' ve ',' have ').replace(' i ',' I ').replace('.',' .').replace('-',' ').replace('  ',' ').strip()
                
                if len(sent_val) < 30 and len(sent_val) > 250 and '(' not in sent_val and ')' not in sent_val:
                    continue
                    
                tok,tag = tag_pos(sent_val)
                tok[0] = tok[0].title()
                
                sent_len = len(sent_val.split())                       
                
                if len(tok) != sent_len:
                    continue
                    
                lbl = []
                for i in range(len(tok)):
                    lbl.append('N')
                
                distinct_mentions = 1
                for dish_val in dish:
                    distinct_mentions *= sent_val.lower().count(dish_val.lower())
                    if dish_val.lower() not in whitelist:
                        distinct_mentions +=1    
                if distinct_mentions > 1:
                    continue
                        
                for dish_val in dish:
                                                                
                    k,l = find_sub_list(dish_val.split(),sent_val.split())
                    if k>=0:
                        for j in range(sent_len):
                            if j >= k and j < k+l:
                                lbl[j] = 'Y'

                pos_lst.append(list(zip(tok,tag,lbl)))
                                
                x+=1
                if x%1000 == 0:
                    printTS(f'Took {x} sentences in {text} texts')
                    
    printTS(f'Took {x} sentences in total')
                       
    return pos_lst
    
def get_ner_dishes(text,name=[],city=[],blacklist=[],whitelist=[]):
        
    # Prepare data for NER
    data = prep_text(text)
    
    # Process with NER and CRF Whitelist            
    return find_dishes_with_ner(data,name=name,city=city,blacklist=blacklist,whitelist=whitelist,startnum=0)
    


if __name__ == "__main__":
    sample_size = 1000000
    
    reviews = load_review_data()    
    doc_dict = process_review_data(reviews.iloc[:sample_size],startnum=0)    
        
    #dump_pickle(doc_dict,f'data/doc_dict.pickle')
    #doc_dict = read_pickle('data/doc_dict.pickle')
    
    pos_tags = tag_pos_from_dish_dict(doc_dict)
    #dump_pickle(doc_dict,f'data/pos_tags.pickle')
    
    print(pos_tags)
    
    print(get_ner_dishes(['I loved the chicken grills but hated the mushroom salad. The Chicken Tikka Masala with Naan was awful.'],whitelist=load_dish_whitelist()))
#END

    
           