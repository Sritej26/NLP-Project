# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:27:47 2021

@author: Sritej. N
"""

import re
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.corpus import brown
from nltk.tag.perceptron import PerceptronTagger
from wordcloud import WordCloud #all the libraries that are being used
from collections import Counter
from pprint import pprint
import en_core_web_sm
import operator
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
#importing the text from text file.
T = open('nlptext3.txt', encoding = 'utf-8', errors = 'ignore').read()
T_strip = T.strip(' ') #strip removes leading and trailing characters based on argument
T = T_strip.replace('\n\n','\n')
T = T.replace('\n\n','\n')
T = T.replace('\t',' ') #pre-processing steps
text = re.sub('\*','',T)
text= re.sub('_','',text)
text = text.lower()
text =re.sub(r"chapter","",text)
text=re.sub(r"[^\w][ivx]+\."," ",text ) #removing roman numbers
text=re.sub(r"some disgusted boys","",text )
text=re.sub(r"birds of a feather","",text ) #removing chapter names
text=re.sub(r"lester brigham's idea","",text )
text=re.sub(r"flight and pursuit","",text )
text=re.sub(r" don’s encounter with the tramp ","",text )
text=re.sub(r" about various things ","",text )
text=re.sub(r" a test of courage ","",text )
text = re.sub(r"i'm", "i am", text)
text = re.sub(r"he's", "he is", text)
text = re.sub(r"she's", "she is", text)
text = re.sub(r"it's", "it is", text)
text = re.sub(r"that's", "that is", text)
text = re.sub(r"what's", "what is", text)
text = re.sub(r"where's", "where is", text)
text = re.sub(r"who's","who is",text)
text = re.sub(r"how's","how is",text)
text = re.sub(r"'s","",text)
text = re.sub(r"\'ll", " will", text)
text = re.sub(r"\'re", " are", text)
text = re.sub(r"\'ve", " have", text)
text = re.sub(r"\'re", " are", text)
text = re.sub(r"\'d", " would", text)
text = re.sub(r"won't", "will not", text)
text = re.sub(r"can't", "can not", text)
text = re.sub(r"ain't","am not",text)
text = re.sub(r"\'t"," not",text)
t = re.sub(r"[^\w]", " ", text)
t = re.sub(r"\n" , " ",t)
words = t.split()
char_num=0
for i in words : #To find number of characters in text
char_num+=len(i)
unique_words = {}
for i in words:
if i not in unique_words: #extracting unique_words from text with their frequency
unique_words[i] = 1
else:
unique_words[i] +=1
freq_distr = {}
for i,j in unique_words.items():
if j not in freq_distr: #frequency of word occurences
freq_distr[j] = 1
else:
freq_distr[j] +=1
plt.bar(freq_distr.keys(), freq_distr.values(), color='g')
plt.yscale('log')
plt.xscale('log')
plt.show()
ps = PorterStemmer()
s_words={} #stemming of words using porterStemmer
for i in words:
if ps.stem(i) not in s_words :
s_words[ps.stem(i)]=1
else :
s_words[ps.stem(i)]+=1
f_words={} #extracting wordlengths and their frequency relations
for i,j in s_words.items() :
if len(i) not in f_words:
f_words[len(i)]=j
else :
f_words[len(i)]+=j
plt.bar(f_words.keys(), f_words.values(), color='g')
plt.show()
wordcloud = WordCloud(max_words=1200, #producing a word Cloud
background_color = 'white',
width = 1200,
height = 1000,
).generate_from_frequencies(s_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
swords = open('stop_words.txt', encoding = 'utf-8', errors = 'ignore').read()
swords=swords.replace('\n',' ')
s=swords.split(" ")
nostop_w={}
for i,j in unique_words.items(): #removing stop-words
if i not in s:
nostop_w[i]=j
wordcloud1 = WordCloud(max_words=1200
background_color = 'white',
width = 1200,
height = 1000,
).generate_from_frequencies(nostop_w)
plt.imshow(wordcloud1)
token = word_tokenize(text)
brown_news_tagged = brown.tagged_sents(categories='news', tagset='universal')
brown_train = brown_news_tagged[100:] #training-set
brown_test = brown_news_tagged[:100] #test-set
tagger = PerceptronTagger(load=False) #declaring tagger
tagger.train(brown_train) #and training it over the training set
p=tagger.tag(token)
a=tagger.evaluate(brown_test) #evaluating it gave 97.4% accuracy on test set
counts = Counter( tag for word, tag in p) #obtaining frequency of POS tags
noun={}
n=0
verb={}
v=0
for i in p:
if(i[1]=='NOUN'): #Extracting all the nouns and verbs from text
if i[0] not in noun:
noun[i[0]]=1
n+=1
else:
noun[i[0]]+=1
if(i[1]=='VERB'):
if i[0] not in verb:
verb[i[0]]=1
v+=1
else:
verb[i[0]]+=1
nsense={}
fsense={}
vsense={}
fvsense={}
for i,j in noun.items() :
lname={}
ln='def'
max=0
for synset in wn.synsets(i): #categorizing nouns based on senses in Wordnet
l=synset.lexname()
if l not in lname:
lname[l] =1
if(lname[l]>max):
ln=l
max=lname[l]
else:
lname[l] +=1
if(lname[l]>max):
ln=l
max=lname[l]
nsense[i]=str(ln)
if ln not in fsense:
fsense[ln]=1
else:
fsense[ln]+=1
f_sorted=sorted(fsense.items(),key=operator.itemgetter(1),reverse=True)
top10cat={} #sorting the Categories based on their frequency
c=0
top10nam={}
for i in f_sorted :
if(c<20): #Extracting Top20 most occurring ‘Categories’ for graph plotting
top10cat[c]=i[1]
top10nam[c]=i[0]
c+=1
else:
break
for i,j in verb.items() :
lname={}
ln='def'
max=0 #categorizing verbs based on senses in Wordnet
for synset in wn.synsets(i):
l=synset.lexname()
if l not in lname:
lname[l] =1
if(lname[l]>max):
ln=l
max=lname[l]
else:
lname[l] +=1
if(lname[l]>max):
ln=l
max=lname[l]
vsense[i]=str(ln)
if ln not in fvsense:
fvsense[ln]=1
else:
fvsense[ln]+=1
e={}
t = re.sub(r"[^\w]", " ", T)
sent = sent_tokenize(t)
nlp = en_core_web_sm.load() #Recognising entity types of words
for s in sent:
doc = nlp(s)
for X in doc :
if(X.ent_type_=='PERSON' and X.pos_==’NOUN’):
if X not in e :
e[str(X)]=str(X.ent_type_)
elif(X.ent_type_=='ORG'):
e[str(X)]=str(X.ent_type_)
elif(X.ent_type_=='NORP'):
e[str(X)]=str(X.ent_type_)
elif(X.ent_type_=='GPE' ):
e[str(X)]=str(X.ent_type_ )
elif(X.ent_type_=='LOC' ):
e[str(X)]=str(X.ent_type_)
sub=[]
link=[]
obj=[]
for s in sent:
nlp = en_core_web_sm.load()
doc = nlp(s)
#pprint([(X, X.ent_iob_, X.ent_type_,X.pos_) for X in doc])
s = ''
i=0
l1 = ''
o = ''
adj=0
status=0 #extracting entity relationship method-1
for X in doc :
if(status==0):
if(X.ent_type_!=''): #if entity type is not null and we haven’t seen a verb till now then that
s = s + ' ' + str(X) # entity is added into subject
if(X.pos_=='CCONJ'):
s = s + ' ' + str(X)
if(X.pos_=='VERB' and status==0): #after we saw a verb we are making it as link
status=1 #between sub and obj
#link[i].append(X)
l1 = l1 + ' ' + str(X)
if(status==1):
if(X.ent_type_!=''): #when we see entity after seeing verb then we add it to object
o = o + ' ' + str(X)
if(X.pos_=='ADJ'):
adj=1
elif(adj==1 and (X.pos_=='NOUN'or 'PROPN')):
o= o +' '+str(X)
adj=0
sub.append(s)
link.append(l1)
obj.append(o) #giving out the relationship at the end .But this method won’t
print(s+l1+o) # work well for passive sentences ...so method-2 has been done.
def subtree_matcher(doc): #method-2 for Entity Relationships extraction
subjpass = 0
for i,tok in enumerate(doc):
# find dependency tag that contains the text "subjpass"
if tok.dep_.find("subjpass") == True:
subjpass = 1 #detecting if a sentence passive or not .
#by searching for a word tagged with subpass dependency
x = ''
y = ''
z = '' #if a passive subject is detected then sentence is passive else active
# if subjpass == 1 then sentence is passive
if subjpass == 1:
for i,tok in enumerate(doc):
if tok.dep_.find("subjpass") == True:
y = tok.text
if (tok.dep_=="ROOT"): #making subject as object and object as subject if sentence is passive
z=tok.text #keeping link as word tagged with ROOT dependency
if tok.dep_.endswith("obj") == True:
x = tok.text
else:
for i,tok in enumerate(doc):
if tok.dep_.endswith("subj") == True:
x = tok.text #if sentence is active then just extract subj,obj and Root verb
if (tok.dep_=="ROOT"):
z=tok.text
if tok.dep_.endswith("obj") == True:
y =y+' '+ tok.text
return x,z,y