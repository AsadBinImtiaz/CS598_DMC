import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Util Functions
from util_funcs import *

# Process_data
from process_data import *

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def load_model():
    cat_model = read_pickle(f'model/cat_model.pickle')    
    cat_vocab = read_pickle(f'model/cat_vocab.pickle')
    return cat_model, cat_vocab    

def print_results(y_true, y_pred, classifier_name):
    print("Confusion Matrix for {}:".format(classifier_name))
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nScore: {}".format(round(accuracy_score(y_true, y_pred)*100, 2)))
        
def create_category_model(sample_size=1000000,save_model=True,save_vocab=True):
    
    #reviews = get_preprossed_review_data(sample_size=sample_size)
    doc_dict = read_pickle('data/doc_dict.pickle')
    cuisine_map = read_pickle(f'data/cuisine_map.pickle')
    
    crf_sent_list, crf_dish_list = flat_dish_dict(doc_dict)
    
    reviews = load_review_data()[['categories']].copy()
    reviews['sents'] = [s.lower() for s in crf_sent_list]
    reviews['dishes'] = crf_dish_list
    reviews = reviews[reviews['dishes'].map(len)>0]
    reviews['cuisine'] =[cuisine_map[' / '.join(cat)] for cat in reviews.categories]
    reviews = reviews[reviews['cuisine'].str.len()>0]
         
    x = [' '.join(d) for d in reviews['dishes']]
    y = reviews['cuisine']
    
    printTS(f"Constructing MNB model")
    
    cat_vocab = CountVectorizer(stop_words=stopwords).fit(x)
    x = cat_vocab.transform(x)
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=101)
    
    cat_model = MultinomialNB(alpha=1.2)
    cat_model.fit(x_train, y_train)
    predmnb = cat_model.predict(x_test)
    print_results(y_test, predmnb, "Multinomial Naive Bayes")
    
    
    if save_model:
        dump_pickle(cat_model,f'model/cat_model.pickle')
    
    if save_vocab:
        dump_pickle(cat_vocab,f'model/cat_vocab.pickle')    
    
    return cat_model
    
def predict_category(test_str,cat_model=None,cat_vocab=None):

    if cat_vocab==None or cat_model==None:
        cat_model, cat_vocab = load_model()
    
    text = pre_process_data([test_str],lowercase_data=True)                    
    x_test  = cat_vocab.transform(text)
    predmnb = cat_model.predict(x_test)
    
    printTS(f"Category: {test_str} -> {predmnb}")
    return predmnb[0]    
    
if __name__ == "__main__":
    
    processed_data = get_preprossed_review_data()
    #dump_pickle(processed_data,f'processed_data.pickle')    
    
    create_category_model()
    mnb_model, mnb_vocab = load_model()
    
    mystr= "The burger was really not great."
    predict_category(mystr)
    
    mystr= "The Tikka was not bad."
    predict_category(mystr)
    
    mystr= "I was tacos dish was very uphappy not happy"
    predict_category(mystr)
    
    
    
    