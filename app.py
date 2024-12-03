import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import gensim
import re
import os
import sys
import nltk
import time
import pdfplumber
import docx
from gensim.models import Word2Vec, KeyedVectors
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
Lemmatizer = WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")

# Pickle files 
wrd2vec = pickle.load(open('artifacts\word2vec.pkl','rb'))
Clf_model = joblib.load('artifacts\model.pkl')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Function for data preprocessing
def preprocess_data(text):
    cleantext = re.sub(r'http\S+|www\.\S+', '',text)
    cleantext = re.sub(r'@\S+','',cleantext) 
    cleantext = re.sub(r'#\S+','',cleantext)
    cleantext = re.sub(r'[^\w\s]','',cleantext)
    cleantext = cleantext.lower()
    words = word_tokenize(cleantext)
    words = [Lemmatizer.lemmatize(word,pos='v') for word in words if not word in stopwords.words('english')]
    cleantext = ' '.join(words)
    return cleantext 

def preprocess_data2(text):
  cleantext = re.sub('[^a-zA-Z]',' ',text)
  cleantext = re.sub('\s+',' ',cleantext)
  return cleantext 

# Function for Average word2vec 
def avg_word_2_vec(doc):
  return np.mean([wrd2vec.wv[word] for word in doc.split() if word in wrd2vec.wv.index_to_key],axis=0)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        extracted_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return extracted_text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text

st.title(':red[Resume Classification]',)
resume_file = st.file_uploader('Upload your resume',type=['pdf','docx'])

if st.button('classify'):
   if resume_file is not None:
       file_type = resume_file.name.split('.')[-1]

       if file_type == 'pdf':
            extracted_txt = extract_text_from_pdf(resume_file)
       elif file_type == 'docx':
            extracted_txt = extract_text_from_docx(resume_file)
       else:
            extracted_txt = ""

       if extracted_txt.strip():
            pre = preprocess_data(extracted_txt)
            pre_cleaned = preprocess_data2(pre)
            vec = avg_word_2_vec(pre_cleaned)
            
            prediction = Clf_model.predict([vec])
            with st.spinner('Please wait...'):
                 time.sleep(5)
            if prediction == 0:
                    st.write('This Uploaded Resume is React Developer')
            elif prediction == 1:
                    st.write('This Uploaded Resume is Workday')
            elif prediction == 2:
                    st.write('This Uploaded Resume is Peoplesoft')
            else:
                    st.write('This Uploaded Resume is SQL Developer')
       else:
            st.write('No text found in the uploaded file . Please upload valid PDF or DOCX file')
   else:
        st.write('Please upload your Resume file')


    













