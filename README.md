# CS598_DMC
# Dish Name Miner
## Extract Dish Names And Predict Their Cuisine Types

## 1. Introduction
The `Dish Name Miner` is the name of the Project dor UIUC CS598 Data Mining final project. In this project, **Dish names** were mined from **Yelp reviews** by performing various NLP tasks.

This project is contributed by:

- Asad Bin Imtiaz (aimtiaz2@illinois.edu)

The analysis and findings are part of **Task-7** for Data Mining Capstone course, wherein the Yelp restaurant reviews were explored to extract dish names mentioned in the review for a restaurant and predict the cuisine type. 
To develop this application system, important features about dish names and their context in reviews texts are learned to develop classification logic to efficiently and effectively mine the dish references in these texts. The goal of this task was to come up with appropriate data science specific approaches to identify important features which represent dish names and compare their predictions.
The idea was to build up on Task-4/5, where Seg-Phrase and Top-Mine frequent pattern mining tools were used to min dish names. In this Task, instead of using these tools one of the following unsupervised techniques were used to mine the dishes and their results compared:
1.	Dish mining with Named Entity Recognizer
2.	Dish Mining with Conditional Random Fields
3.	Dish Mining with Word2Vec model.
The resulting applications is deployed at: http://18.157.64.217/
The code repository of this project can be found at: https://github.com/AsadBinImtiaz/CS598_DMC
In the following, the implementation of these tasks and their sub tasks are explained.


### 1.1. Required Libraries

1.1	Tools and Libraries: 
The exploration and analysis of Yelp restaurant dataset was performed with Python 3. Several python libraries such as **gensim**, **sklearn**, **scipy**, **numpy** and **Scikit-learn** were used to analyze and cleanse the review texts.

### 1.2 Method

The extraction of Dish-Names is not an easy task, and there was no existing work found for directly mining dish names from text data. The dish mentions are hard to separate from other nouns. It is not easy to separate Harry Potter and Pot Pie, or Sweet Person and Sweet Potato, or Turkish Grill (restaurant name) vs Turkey Grill (dish name). On top, there can be spelling variations and mistakes (like Kebab and Kabab), or abbreviations (like veggie roll vs vegetable roll), or non-capitalizations in dish names (such as red chicken vs Red Chicken) which could make NLP library wrongly POS tag words (e.g red as adjectives instead of noun in Red Chicken). Moreover, many dish names have conjunction words such as With or And in them, such as in 'Mac And Cheese', or in a dish name like 'Thai Beef Curry With Rice'. Therefore it was important to ensemble several approaches to effectiveluy min dish names. 
For this task, following approaches were explored to extract dish names: 
1.	Using Named-Entity-Recognition (NER) with POS tagging
2.	Using Conditional-Random-Fields (CRF) to classify words are dishes
3.	Using Word2Vec model to find similarity of phrases with words like 'Dish', 'Food', 'Cuisine' etc.

### 1.3 Installation

Following python libraries are needed:

pip3 install flask
pip3 install pandas
pip3 install spacy
pip3 install re
pip3 install pickle
pip3 install itertools
pip3 install gensim
pip3 install fuzzywuzzy
pip3 install python-levenshtein
pip3 install django
pip3 install sklearn_crfsuite

python -m spacy download en_core_web_sm

### 1.4 Running the WebApp

run `python3 app.py` 

Wait for web app to start.
In browser, load the website: http://127.0.0.1

## 7. Project structure

**src** folder contains all webapp source code
**template** folder contains all html flask templates
**data** folder contains raw dataset downloaded
**models** contains all pickeld classifiers and models

Data Mining Capstone Final Project
