import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import string
import emoji

nltk.download('punkt_tab')
stopw = set(nltk.corpus.stopwords.words("spanish")) # change to language of choice

df = pd.read_csv('data.csv') #load you data

web_re = re.compile(r"https?:\/\/[^\s]+", re.U) #remove webistes
user_re = re.compile(r"(@\w+\-?(?:\w+)?)", re.U) # remove users
hashtag_re = re.compile(r"(#\w+\-?(?:\w+)?)", re.U) # remove '#xxx'
punctuation_re = re.compile(f"[{re.escape(string.punctuation)}]")  # remove punctuation
number_re = re.compile(r"\d+")  # Matches any sequence of digits

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="") 

stopw = set(nltk.corpus.stopwords.words("spanish"))

def preprocess(text):
    text = str(text)
    text = web_re.sub("", text)  
    text = user_re.sub("", text)  
    text = hashtag_re.sub("", text)  
    text = number_re.sub("", text)
    text = punctuation_re.sub("", text) 
    text = remove_emojis(text)
    text = text.lower()  
    return text



# Apply the tokenize function to any text to return a text that has been cleaned and had stopwords removed 
def tokenize(cell):
    sentence = preprocess(cell)
    tokens = word_tokenize(sentence, language='spanish')  # Tokenize correctly
    filtered_tokens = [word for word in tokens if word not in stopw]  # Remove stopwords
    return ' '.join(filtered_tokens)

