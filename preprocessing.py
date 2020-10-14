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
  

case_docs=[]

#Preprocess case docs!
for i in range(1,2195):
	file=open("./case_docs/C"+str(i)+".txt")
	case_lines=file.readlines()
	cleaned_doc=[]
	for line in case_lines:
		#text preprocessing for indidual doc
		line=line.lower() 
		line=remove_whitespace(line)
		line=remove_punctuation(line)
		line=remove_stopwords_and_stem(line)
		cleaned_doc.append(' '.join(line))
	case_docs.append({"id":i,"doc_text":cleaned_doc})


#Write JSON object to file
with open('case_docs.json', 'w') as f:
    json.dump(case_docs, f)




	



