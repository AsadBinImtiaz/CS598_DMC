#!/usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *
from crf_model import *
from ner_pos_tagging import *
from process_data import *
from category_model import *
from word2vec_model import *

reviews = load_review_data()[['review_id','text','name','city']].copy()
name    = pre_process_data(reviews.name.tolist(),lowercase_data=True,remove_newlines=False)
city    = pre_process_data(reviews.city.tolist(),lowercase_data=True,remove_newlines=False)

cat_model      = read_pickle(f'model/cat_model.pickle')    
cat_vocab      = read_pickle(f'model/cat_vocab.pickle')
crf_model      = read_pickle('model/crf_model.pickle')
w2m_model      = read_pickle('model/'+w2v_model_name+'.pickle')

def load_select_list_items():
    return read_pickle('data/select_list.pickle')     

def get_result_body(rev_text,crf_dict,ner_dict,w2v_dict):
    str_txt = get_result_body_template();
    str_txt = str_txt.replace('strRestaurantReview',str(rev_text).replace('\\','').replace('\\n','<br/>').replace('\n','<br/>'))
    str_txt = str_txt.replace('<#CRF#>',get_sent_html(crf_dict))
    str_txt = str_txt.replace('<#CRF_NER#>',get_sent_html(ner_dict))
    str_txt = str_txt.replace('<#W2V#>',get_sent_html(w2v_dict))
    return str_txt
    
def get_results_body(text):
    
    processed_text = pre_process_data(text,lowercase_data=False)
    
    dish_names, dish_dict =find_dishes_with_crf(crf_model,processed_text)
    printTS(f"CRF Found : {dish_names}")
    crf_sent_list, crf_dish_list = flat_dish_dict(dish_dict)
    
    crf_dict  = replace_best_matches(dish_dict,prefix="</span><span style='color:#293795'><b>",postfix='</b></span>')
    
    dish_dict = get_ner_dishes(processed_text,city=city,name=name,blacklist=global_BL,whitelist=dish_names)
    
    ner_dict  = replace_best_matches(dish_dict,prefix="</span><span style='color:#293795'><b>",postfix='</b></span><span>')
    
    ner_sent_list, ner_dish_list = flat_dish_dict(ner_dict)
    printTS(f"NER+CRF Found : {ner_dish_list}")
    
    dish_dict = find_wv_dish(text,model=w2m_model,blacklist=global_BL)
    w2v_dict  = replace_best_matches(dish_dict,prefix="</span><span style='color:#293795'><b>",postfix='</b></span><span>')
    v2w_sent_list, w2v_dish_list = flat_dish_dict(w2v_dict)
    printTS(f"W2V Found : {w2v_dish_list}")
    
    review_text = replace_best_matches_in_text(text[0],ner_dish_list[0],prefix="</span><span style='color:#293795'><b>",postfix='</b></span><span>',use_orig=True)
    str_txt = get_result_body(review_text,crf_dict,ner_dict,w2v_dict)
        
    return str_txt

def get_result_body_analyse(str_rev):
    printTS(f"Called: get_result_body_analyse called: {str_rev}")
    
    revs = reviews[reviews.review_id==str_rev]
    text = revs.text.tolist()
    
    return get_results_body(text)
        
def get_result_body_play(str_txt=""):
    printTS(f"Called: get_result_body_play called: Len={len(str_txt)}")
    
    return get_results_body([str_txt])
    
#def get_result_body_filled():
def get_result_body_template():
    return """
                            <div style="width: 100%">
                                <table style='width:100%'>
                                    <tr>
                                        <td valign=top style='padding:0px 18px'>
                                        <table style='width:100%'>
                                            <tr>
											<td style="width: 15%; vertical-align: top;">
												<label style="text-align: left; vertical-align: top;"><b>Review Text:</b></label>
                                                <br/>
                                                <span style='font-size:9px;color:#293795;'>(Mined dishes are highlighted)</span>
                                                <br/>
											</td>
											<td style="width 85%; margin-right: 15%;">
												<label style="text-align: left; margin-right: 15%;"><table><tr>strRestaurantReview</tr></table></label>
											</td>
											</td>
											</tr>
										</table>
										</td>
									</tr>
                                    <tr>
                                        <td><p><b><span>&nbsp;</span></b></p></td>
                                    </tr>
                                    <tr>
                                        <td valign=top style='padding:0px 20px'>
                                            <p><b><u><span style='font-size:16px;color:#222333;'>Analysis Results:</span></u></b></p>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><p><b><span>&nbsp;</span></b></p></td>
                                    </tr>
                                    <tr>
                                        <td valign=top style='padding:0px 10px'>
                                            <table style='width:100%;'>
                                                <tr>
                                                    <td valign=top style='padding:0px 10px'>
                                                        <p><span style='font-size:17px;color:#FF5733;'><u><i>Dish names using Named Entity Recognition (NER) with Conditional random fields (CRFs):</i></u></span><u></p>
                                                    </td>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                </tr>
                                                <tr style='background:#C5E0B3'>
                                                    <td valign=top style='width:60%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Review Sentence</span></b></p>
                                                    </td>
                                                    <td valign=top style='width:20%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Mined Dist Name(s)</span></b></p>
                                                    </td>
                                                    <td valign=top style='width:20%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Predicted Cuisine</span></b></p>
                                                    </td>
                                                </tr>
                                                <#CRF_NER#>
                                            </table>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td valign=top style='padding:0px 10px'>
                                            <table style='width:100%'>
                                                <tr>
                                                    <td valign=top style='padding:0px 10px'>
                                                        <p><span style='font-size:17px;color:#FF5733;'><u><i>Dish names using Conditional random fields (CRFs) modelling:</i></u></span><u></p>
                                                    </td>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                </tr>
                                                <tr style='background:#BDD6EE'>
                                                    <td valign=top style='width:60%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Review Sentence</span></b></p>
                                                    </td>
                                                    <td valign=top style='width:20%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Mined Dist Name(s)</span></b></p>
                                                    </td>
                                                    <td valign=top style='width:20%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Predicted Cuisine</span></b></p>
                                                    </td>
                                                </tr>
                                                <#CRF#>
                                            </table>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td valign=top style='padding:0px 10px'>
                                            <table style='width:100%'>
                                                <tr>
                                                    <td valign=top style='padding:0px 10px'>
                                                        <p><span style='font-size:17px;color:#FF5733;'><u><i>Dish names using Word2Vec:</i></u></span><u></p>
                                                    </td>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                </tr>
                                                <tr style='background:#BDD6EE'>
                                                    <td valign=top style='width:60%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Review Sentence</span></b></p>
                                                    </td>
                                                    <td valign=top style='width:20%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Mined Dist Name(s)</span></b></p>
                                                    </td>
                                                    <td valign=top style='width:20%'>
                                                        <p><b><span style='font-size:16px;padding:0px 10px;'>Predicted Cuisine</span></b></p>
                                                    </td>
                                                </tr>
                                                <#W2V#>
                                            </table>
                                        </td>
                                    </tr>
                                </table>							    
							</div>
							"""

def get_sent_html(dish_dict):
    sent_dict = dish_dict[0]
    ret_html = ""
    str_html = """
                                                <tr style='width:100%'>
                                                    <td valign=top style='padding:0px 10px;width:60%'>
                                                        <p><span><#SENTENCES#></span></p>
                                                    </td>
                                                    <td valign=top style='padding:0px 10px'>
                                                    <table style='width:100%'>
                                                        <#DISHLIST#>
                                                    </table>
                                                    <td valign=top style='padding:0px 10px;width:60%'>
                                                        <p><span><#CATEGORY#></span></p>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td><p><b><span>&nbsp;</span></b></p></td>
                                                </tr>
                                                """
    if len(sent_dict) == 0:
        return str_html.replace('<#SENTENCES#>','<i>No dish fount in review text<i>').replace('<#DISHLIST#>','').replace('<#CATEGORY#>','')                                                
    for sent,dishes in sent_dict.items():
        dishes = list(set(dishes))
        for dish in dishes:
            if dish+'s' in dishes:
                dishes.remove(dish)
        sent_html = str_html  
        sent_html = sent_html.replace('<#SENTENCES#>',sent.replace(' ,',',').replace(' .','.').replace('\n','<br>'))
        dish_tbl = ""
        for dish in dishes:
            dish_tbl = dish_tbl + '<tr><td><p><b><span style="color:#800080">'+dish.title()+'</span></b></p></td></tr>\n'    
        sent_html = sent_html.replace('<#DISHLIST#>',dish_tbl)    
        cat = predict_category(' '.join(dishes),cat_model=cat_model,cat_vocab=cat_vocab)
        sent_html = sent_html.replace('<#CATEGORY#>',cat)    
        ret_html += sent_html
        
    return ret_html

if __name__ == "__main__":
    start_logger()
    printTS (len(load_select_list_items()))