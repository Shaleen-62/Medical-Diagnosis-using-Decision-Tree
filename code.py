from pandas.core.arrays import string_
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
import argparse
import warnings
warnings.simplefilter("ignore")
import pickle
import re
import csv
from googlesearch import search
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import requests
from bs4 import BeautifulSoup
import math
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from time import time
from itertools import combinations
from collections import Counter
import operator
import nltk

nltk.download('all')

# to return a synonym list of the word given by user
def extracting_synonyms(word):
  l=list()
  page = requests.get('https://www.thesaurus.com/browse/{}'.format(word))
  html_content = BeautifulSoup(page.content, "html.parser")
  try:
    container = html_content.find('section', {'class': 'MainContentContainer'})
    card = container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
    card = card.find_all('li')
    for i in card:
      l.append(i.get_text())
  except:
    None
  for i in wordnet.synsets(word):
    l+=i.lemma_names()
  l=set(l)
  return l
 
#gives description of disease after searching the content from wikipedia
def brief(data):
  term=[data]
  answer= data+" \n"
  for i in term:
    #making the query term to search for disease's wikipedia information in google
    query = i+' wikipedia'
    for j in search(query,tld="co.in",stop=10,pause=0.5):
      find=re.search(r'wikipedia',j)
      flag=0
      if find:
        page=requests.get(j,verify=False)
        html_content = BeautifulSoup(page.content, "html.parser")
        info = html_content.find("table", {"class":"infobox"})
        if info is not None:
          for k in info.find_all("tr"):
            information=k.find("th",{"scope":"row"})
            if information is not None:
              s=str(k.find("td"))
              s=s.replace('.','')
              s=s.replace(';',',')
              s=s.replace('<b>','<b> \n')
              s=re.sub(r'<a.*?>','',s) #removal of hyperlinks
              s=re.sub(r'</a>','',s) #removal of hyperlinks
              s=re.sub(r'<[^<]+?>',' ',s) #removal of tags
              s=re.sub(r'\[.*\]','',s) #removal of citation text
              s=s.replace("&gt",">")
              #appending the information to the answer
              answer+=information.get_text()+" - "+s+"\n"
              flag=1
        if flag:
          break
  return answer

def main():
  stop_words = stopwords.words('english') # english words which doesn't add much meaning to a sentence
  lemmatized_words = WordNetLemmatizer() # morphological analysis of words grouping them as a single item
  splitting_words = RegexpTokenizer(r'\w+') # splitting string into substrings and creating tokens 

  dataframe1=pd.read_csv("Dataset/disease_symptom_dataset1.csv")#Dataset containing combination of symptoms of each disease
  dataframe2=pd.read_csv("Dataset/disease_symptom_dataset2.csv")#Dataset containing single row for each disease having all the symptoms maintained as boolean values.

  # training the disease_symptom dataset using decision tree classifier
  X_train = dataframe1.iloc[:, 1:]
  Y_train = dataframe1.iloc[:, 0:1]

  Tree = DecisionTreeClassifier()
  Tree = Tree.fit(X_train, Y_train)
 
  # storing all the symptoms in the dataset as a list 
  symptom_dataset = list(X_train.columns)

  # input from the user(symptoms)
  inputs = input_group("Disease Prediction",[
                                             input("Enter synonyms separated by comma and spacebar:", name='user_input', type=TEXT),
  ])                                           
  inputs["user_input"] = inputs["user_input"].lower().split(', ')
  user_symptoms=[]
  # processing the user input
  for i in inputs["user_input"]:
    i=i.strip()
    i=i.replace('-', ' ')
    i=i.replace("'",'')
    i=''.join([lemmatized_words.lemmatize(word) for word in splitting_words.tokenize(i)])
    user_symptoms.append(i)

  # searching for the synonyms of the user symptoms and attaching it to the user input list
  symptoms = []
  for i in user_symptoms:
    i = i.split()
    j=set()
    for k in range(1, len(i)+1):
      for sets in combinations(i,k):
        sets = ' '.join(sets)
        sets = extracting_synonyms(sets)
        j.update(sets)
    j.add(' '.join(i))
    symptoms.append(' '.join(j).replace('_',' '))

  # calculating similarity score(jacquard coefficient) of the symptoms in dataset and the ones entered by user.
  final_symptoms=set()
  threshold = 0.5
  for i, j in enumerate(symptom_dataset):
    k = j.split()
    for sym in symptoms:
      c=0
      for k1 in k:
        if k1 in sym.split():
          c+=1
      if c/len(k)>threshold:
        final_symptoms.add(j)
  put_text("Final matching symptoms from your input: ")
  for i,j in enumerate(list(final_symptoms)):
    put_text(i,":",j)
  #finding other relevant symptoms according to user input by co-occurance method
  number_list = input("\n Select the ids of relevant symptoms space separated: \n", type=TEXT).split()
  disease = set()
  final_final_symptoms=[]
  cnt=[]
  for i in number_list:
    j = list(final_symptoms)[int(i)]
    final_final_symptoms.append(j)
    disease.update(set(dataframe2[dataframe2[j]==1]['label_dis']))
  for d in disease:
    dis_r = dataframe2.loc[dataframe2['label_dis'] == d].values.tolist()
    dis_r[0].pop(0)
    for id,value in enumerate(dis_r[0]):
      if value!=0 and symptom_dataset[id] not in final_final_symptoms:
        cnt.append(symptom_dataset[id])
  # finding the co-occuring symptoms of the user input
  symp=dict(Counter(cnt))
  symptom_tuple = sorted(symp.items(), key=operator.itemgetter(1), reverse=True)
  extracted = []
  count=0
  for t in symptom_tuple:
    count+=1
    extracted.append(t[0])
    if count%5 == 0 or count==len(symptom_tuple):
      put_text("\nCo-occuring symptoms are: ")
      for id,element in enumerate(extracted):
        put_text(id,":",element)
      #asking the user that whether he is experiencing these co-occuring symptoms of its input
      number_list = input("Are you experiencing any of these symptoms. If yes, enter the ids space separated, 'no' to stop, '-1' to skip and get next set of symptoms: \n", type=TEXT).split()
      if(number_list[0]=='no'):
        break
      if(number_list[0]=='-1'):
        extracted=[]
        continue
      for i in number_list:
        final_final_symptoms.append(extracted[int(i)])
      extracted=[]

  # creating the final list of symtoms and setting it as the test data
  put_text("\n Final List of symptoms: ")
  X_test = [0 for i in range(0, len(symptom_dataset))]
  for j in final_final_symptoms:
    put_text(j)
    X_test[symptom_dataset.index(j)]=1

  #predicting disease using a decision tree classifier and showing the top 5 diseases with their probabilities
  dis = Tree.predict_proba([X_test])
  k=5
  predicted_diseases=list(set(Y_train['label_dis']))
  predicted_diseases.sort()
  top=dis[0].argsort()[-k:][::-1]

  put_text("\n Top 5 diseases are:")
  topk={}
  for i,j in enumerate(top):
    symp=set()
    #extracting the whole row of that specific disease
    dis_row=dataframe2.loc[dataframe2['label_dis']==predicted_diseases[j]].values.tolist()
    #extracting the binary vector of symptoms of that disease
    dis_row[0].pop(0)
    for i,l in enumerate(dis_row[0]):
      if(l!=0):
        symp.add(symptom_dataset[i])
    #calculating the match score of user input binary vector with the actual symptom binary vector
    match_score=(len(symp.intersection(set(final_final_symptoms)))+1)/(len(set(final_final_symptoms))+1)
    #calculating accuracy of X_test with each disease's symptom binary vector
    accuracy = accuracy_score(dis_row[0], X_test)
    #calculating probability of each disease
    probability = match_score*accuracy
    topk[j]=probability
  cnt=0
  b={}
  #sorting the probabilities of diseases
  sort = dict(sorted(topk.items(), key=lambda x: x[1], reverse=True))
  for i in sort:
    probability = sort[i]*100 #probability percentage
    put_text(str(cnt) + predicted_diseases[i], " Probability: ", str(round(probability, 2)) + "%")
    b[cnt]=i
    cnt+=1
    
  number=input("\nIf you want to get a diagnosis, enter the index of disease on which you want to get a diagnosis, else insert -1 to stop the system:\n", type=TEXT)
  if number!='-1':
    word=predicted_diseases[b[int(number)]]
    put_text()
    put_text(brief(word))
    put_text("If you are facing serious issues, contact your nearest doctor to get an accurate diagnosis.")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--port", type=int, default=8000)
  parser.add_argument('-f')
  args = parser.parse_args()

  start_server(main, port=args.port)

