import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import streamlit as st
#Tfidf_vect = TfidfVectorizer(ngram_range=(1,2),max_features=1000)
#Tfidf_vect = pickle.load(open("C:\Windows\System32\Intern\model_TF-IDF.sav", 'rb'))
#Stemming
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['headlines']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

#Lemmatization
lemmatizer = WordNetLemmatizer()
def lem_list(row):
    my_list = row['headlines']
    lemmatized_list = [lemmatizer.lemmatize(word) for word in my_list]
    return (lemmatized_list)

#Removing stop words
stops = set(stopwords.words("english"))
def remove_stops(row):
    my_list = row['headlines']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

#Rejoining Tweets
def rejoin_words(row):
    my_list = row['headlines']
    joined_words = ( " ".join(my_list))
    return joined_words

def data_cleaning(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text=nltk.word_tokenize(text)
    text=[stemming.stem(word) for word in text]
    text=[lemmatizer.lemmatize(word) for word in text]
    text=[w for w in text if not w in stops]
    text=( " ".join(text))
    return text
def fake_headlines_test(news):
    new_headlines={"text":[news]}
    new_headlines_test = pd.DataFrame(new_headlines)
    new_headlines_test["text"] = new_headlines_test["text"].apply(data_cleaning)
    new_headlines_test = new_headlines_test["text"]
    Tfidf_vect = pickle.load(open("C:\Windows\System32\Intern\model_TF-IDF.sav", 'rb'))
    new_headlines_vectors_test=Tfidf_vect.transform(new_headlines_test)
    loaded_model_Passive = pickle.load(open("C:\Windows\System32\Intern\Passive_finalized_model.sav", 'rb'))
    #outcome_MNB=loaded_model.predict(new_headlines_vectors_test)
    outcome_passive_Reg=loaded_model_Passive.predict(new_headlines_vectors_test)
    if outcome_passive_Reg==1:
      return "Real News"
    else:
      return "Fake News"
def main():
    st.title("Covid-19 Headlines Span Detection")
    sentence = st.text_input('Input your sentence here:')
    if sentence:
        result=fake_headlines_test(sentence)
    results=st.button("Predict")
    if results:
        st.write(result)





if __name__=='__main__':
    main()
