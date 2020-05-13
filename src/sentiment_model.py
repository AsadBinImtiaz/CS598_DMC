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


mnb_model = None
mnb_vocab = None

def load_model():
    mnb_model = read_pickle(f'model/mnb_model.pickle')    
    mnb_vocab = read_pickle(f'model/mnb_vocab.pickle')
    return mnb_model, mnb_vocab    

def print_results(y_true, y_pred, classifier_name):
    print("Confusion Matrix for {}:".format(classifier_name))
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nScore: {}".format(round(accuracy_score(y_true, y_pred)*100, 2)))

def process_sentiment_text(data):
    new_data = []
    i = 0
    x = len(data)

    printTS('Start processing sentiment text')
    for doc in nlp.pipe(iter(data), batch_size=1000, n_threads=20):
        tokens = []
        sentis = []
        skip = False
        
        if len(new_data) % 10000 == 0:
            printTS(f"Processed: {len(new_data)}/{x}")
        
        for sent in doc.sents:
            sent_words = []
            for i,token in enumerate(sent):
                if skip:
                    skip = False
                    
                else:
                    lemma = token.lemma_.strip()
                    word  = token.text
                    pos   = token.pos_
                    
                    if token.is_stop:
                        skip = False
                        continue
                    
                    if (token.pos_ in sentiment_pos or token.text in negations) and len(token.lemma_) > 1:
                        if i+1<len(sent) and pos in ['ADJ'] and sent[i+1].pos_ in ['NOUN'] and '_' not in sent[i+1].lemma_:
                            sent_words.append(lemma+"-"+sent[i+1].lemma_)
                            skip = True
                        elif i+1<len(sent) and lemma in negations and sent[i+1].pos_ in ['ADJ', 'ADV'] and sent[i+1].lemma_ not in negations:
                            sent_words.append(lemma+"-"+sent[i+1].lemma_)
                            skip = True
                        elif len(lemma.replace(".",""))>1:
                            sent_words.append(lemma.replace(" ","-"))
            
            sentis.append(" ".join(sent_words).replace(".",""))

        text = str(" ".join((value for value in sentis if value != '.'))).replace("_","-") #.replace("-","_")
        
        if len(text.split())<=3 and '-' not in text:
            text = doc.text
        new_data.append(text)
        
        print(text)
        i+=1
    
    printTS('Sentiment data processed')
    
    return new_data
    
def get_preprossed_sent_text(text):
    processed_data = pre_process_data([text],lowercase_data=True)
    return processed_data
    
def get_preprossed_review_data(sample_size=1000000):
    reviews = load_review_data().iloc[:sample_size]
    processed_data = pre_process_data(reviews['text'],lowercase_data=True)
    reviews['processed_text'] = process_sentiment_text(processed_data)
    
    return reviews
        
def create_sentiment_model(sample_size=1000000,save_model=True,save_vocab=True):
    
    #reviews = get_preprossed_review_data(sample_size=sample_size)
    reviews = read_pickle('processed_data.pickle')
    
    reviews = reviews[(reviews['stars']==1) | (reviews['stars']==2) | (reviews['stars']==4) |(reviews['stars']==5)]
    x = reviews['processed_text']
    y = reviews['stars']
    
    #x = [xx.replace("-","_") for xx in x]
    
    y.replace([1,2,3,4,5],['Neg','Neg','Neu','Pos','Pos'], inplace=True)
    
    mnb_vocab = CountVectorizer(stop_words=stopwords).fit(x)
    x = mnb_vocab.transform(x)
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=101)
    
    mnb_model = MultinomialNB(alpha=1.2)
    mnb_model.fit(x_train, y_train)
    predmnb = mnb_model.predict(x_test)
    print_results(y_test, predmnb, "Multinomial Naive Bayes")
    
    rmfr = RandomForestClassifier(n_estimators=500)
    rmfr.fit(x_train, y_train)
    p = rmfr.predict(x_test)
    print_results(y_test, p, "Random Forest Classifier")

    mlp = MLPClassifier(max_iter=5000)
    mlp.fit(x_train,y_train)
    p = mlp.predict(x_test)
    print_results(y_test, p, "Multilayer Perceptron")
    
    mnb_model = mlp
    
    if save_model:
        dump_pickle(mnb_model,f'model/mnb_model.pickle')
    
    if save_vocab:
        dump_pickle(mnb_vocab,f'model/mnb_vocab.pickle')    
    
    return mnb_model
    
def predict_sentiment(test_str):
    
    mnb_model, mnb_vocab = load_model()

    text = process_sentiment_text(get_preprossed_sent_text(test_str))                    
    x_test  = mnb_vocab.transform(text)
    predmnb = mnb_model.predict(x_test)
    
    printTS(f"Sentiment: {test_str} -> {predmnb}")
    return predmnb[0]    
    
if __name__ == "__main__":
    
    #processed_data = get_preprossed_review_data()
    #dump_pickle(processed_data,f'processed_data.pickle')    
    
    #create_sentiment_model()
    mnb_model, mnb_vocab = load_model()
    
    mystr= "The  was really not great."
    predict_sentiment(mystr)
    
    mystr= "The was not bad."
    predict_sentiment(mystr)
    
    mystr= "I was dish was very uphappy not happy"
    predict_sentiment(mystr)
    
    #mystr= "The dish was bad"
    #predict_sentiment(mystr)
    
    