from collections import defaultdict,Counter
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
import json

stemmer = SnowballStemmer("english")




def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator)

def remove_whitespace(text): 
    return  " ".join(text.split()) 


def remove_stopwords_and_stem(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    stems = [stemmer.stem(word) for word in filtered_text] 
    return stems
  


N=2914
#Preprocess case docs!
DF=defaultdict(set)
doc_count=defaultdict()
doc_tokens=defaultdict()
for i in range(1,N+1):
    file=open("./case_docs/C"+str(i)+".txt")
    case_lines=file.readlines()
    cleaned_doc=[]
    for line in case_lines:
        #text preprocessing for indidual doc
        line=line.lower() 
        line=remove_whitespace(line)
        line=remove_punctuation(line)
        tokens=remove_stopwords_and_stem(line)
        doc_count[i]=Counter(tokens) #Term Frequency!
        #doc frequency
        for w in tokens:
            DF[w].add(i) #doc id
        
        doc_tokens[i]=tokens


for w in DF:
    DF[w]=len(DF[w]) #we only need the count and not the doc indices! 

#Iterate over the docs again and for every token in a doc calculate tf_idf
tf_idf={}
case_docs=[]
for i in range(1,N+1):
    tokens=doc_tokens[i]
    counter=doc_count[i]
    words_count=len(counter)
    doc_tfidf=[] 
    for token in np.unique(tokens):
        tf = counter[token]/words_count
        df = DF[token]
        idf = np.log(N/(df+1))
        tf_idf[i, token] = tf*idf  #doc,
        doc_tfidf.append(tf*idf)

    doc_tfidf=sorted(doc_tfidf,reverse=True)
    case_docs.append({"id":i,"tf_idf":doc_tfidf})




#Write JSON object to file
with open('case_docs.json', 'w') as f:
    json.dump(case_docs, f)






    



