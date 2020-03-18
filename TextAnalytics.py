#!/usr/bin/env python
# coding: utf-8

# # Text Analytics

# #### Charlie Marshall
# #### Prof. Klabjan
# #### IEMS 308
# #### 2 March 2020

# In[2]:


import pandas as pd
import numpy as np
import scipy
import re
import glob
import os
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from nltk import pos_tag
import spacy

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

from imblearn.over_sampling import SMOTE
import statsmodels.api as sm


# ### Load in Data

# In[3]:


percent = pd.read_csv("/Users/charlesmarshall/Desktop/IEMS 308/Project 3/all/percentage.csv", engine = "python", names = ['perc'])


# In[4]:


percent.head()


# In[5]:


ceo = pd.read_csv("/Users/charlesmarshall/Desktop/IEMS 308/Project 3/all/ceo.csv", engine = "python", names = ['first', 'last'])


# In[6]:


def ceo_name(df):
    for i in range(len(ceo)):
        if pd.isnull(ceo.loc[i,'last']):
            ceo.loc[i,'ceo_full'] = ceo.loc[i,'first']
        elif pd.isnull(ceo.loc[i,'first']):
            ceo.loc[i,'ceo_full'] = ceo.loc[i,'last']
        else:
            ceo.loc[i,'ceo_full'] = ceo.loc[i,'first'] + ' ' + ceo.loc[i,'last']
            
    return df;


# In[7]:


ceo = ceo_name(ceo)


# In[8]:


ceo = ceo.drop(['first','last'], axis=1)


# In[9]:


ceo.head()


# In[10]:


company = pd.read_csv("/Users/charlesmarshall/Desktop/IEMS 308/Project 3/all/companies.csv", engine = "python", names = ['company'])


# In[11]:


company.head()


# In[12]:


file_list = glob.glob("/Users/charlesmarshall/Desktop/IEMS 308/Project 3/*/*.txt")

corpus = []

for file_path in file_list:
    with open(file_path,encoding='ISO-8859-1') as f_input:
        corpus.append(f_input.read())


# In[13]:


len(corpus)


# ## Clean data
# 
# Remove all unicode and *.

# In[14]:


print(corpus[0])


# In[15]:


for text in range(len(corpus)):
    corpus[text] = re.sub(r'[^\x00-\x7f]|[*]',r'', corpus[text])


# In[16]:


print(corpus[0])


# ## Tokenizing the Sentences

# In[17]:


sentences = []

for text in range(len(corpus)):
    s = sent_tokenize(corpus[text])
    sentences.append(s)


# In[18]:


len(sentences)


# In[19]:


sentences = [item for sublist in sentences for item in sublist]


# In[20]:


len(sentences)


# In[21]:


sentences[6940]


# ### Removing stop words in sentences:
# 
# None of the categories we are looking for (CEOs, percentages, or Companies) should include stop words,
# so  removing them will not eliminate any candidates which simulataneously eliminating candidates which do not deserve to be picked

# In[22]:


stop_words=sorted(set(stopwords.words("english")))


# In[23]:


def drop_stop_words(ls):
    for i in range(len(ls)):
        tokenized_sent = word_tokenize(ls[i])
        ls[i] = ' '.join([word for word in tokenized_sent if word.lower() not in stop_words])
        
    return ls;


# In[24]:


sentences = drop_stop_words(sentences)


# In[25]:


sentences[6940]


# ## CEO's

# 1) Find all the names of people included in the corpus (potential CEO's). This is done by searching for any value that has two uppercase words in a row or just one uppercase word. It is not the most exact way to do this (for instance, there are lots of words at the beginning of sentences which are included, but many of these words should be eliminated in feature selection.
# 
# 2) Blocks of text (paragraphs, windows, sentences, etc) will be inspected to come up with features.
# 
# - Potential Features:
# 1) CEO is in the same sentence (should correctly identify people who are obviously CEOs)
# 2) Word/ word phrase is longer than 3 characters (many of the stop words which are included in the potential ceo list are just words which start sentences, but can be eliminated because they have only a few characters)
# 3) I'm not sure - this might be good
# 
# 3) A df will then be created with the row name being the name of each person and each column being a feature. 
# 
# 4) Train a logistic regression model on half of the data
# 
# 5) Test the model on the other half of the data. 

# ### Creating df for classification

# In[26]:


def cap_letters(message):
    caps = sum(1 for c in message if c.isupper())
    return caps;


# In[27]:


def cap_in_sent(ls):
    sent_caps = sum(1 for c in ls if c.isupper())
    return sent_caps;


# In[28]:


def sentence_words(ls):
    ceos = 0
    sens = 0
    pres = 0
    inv = 0
    aut = 0
    represent = 0
    ambass = 0
    secr = 0
    exp = 0
    spok = 0
    gov = 0
    part = 0
    found = 0
    who=0
    
    if re.findall(r'CEO|ceo', ls) != []: 
        ceos = 1
    if re.findall(r'Senator|Sen.', ls) != []: 
        sens = 1
    if re.findall(r'President', ls) != []: 
        pres = 1
    if re.findall(r'investor|Investor', ls) != []: 
        inv = 1
    if re.findall(r'author|Author', ls) != []: 
        aut = 1
    if re.findall(r'Representative|Rep.', ls) != []: 
        represent = 1
    if re.findall(r'Ambassador|ambassador', ls) != []: 
        ambass = 1
    if re.findall(r'Secretary|secretary', ls) != []: 
        secr = 1
    if re.findall(r'Expert|expert', ls) != []: 
        exp = 1
    if re.findall(r'spokesman|spokeswoman|Spokesman|Spokeswoman', ls) != []: 
        spok = 1
    if re.findall(r'Governor|Gov.', ls) != []: 
        gov = 1
    if re.findall(r'partner|Partner', ls) != []: 
        part = 1
    if re.findall(r'founder|Founder', ls) != []:
        found = 1
    if re.findall(r'who|Who', ls) != []:
        who = 1
        
    return ceos, sens, pres, inv, aut, represent, ambass, secr, exp, spok, gov, part, found,who;


# In[29]:


def person_two_before(sent,phrase_in_sent):
    try:
        who_two_before = 0
        ceo_two_before = 0
        sen_two_before = 0
        pres_two_before = 0
        inv_two_before = 0
        aut_two_before = 0
        rep_two_before = 0
        amb_two_before = 0
        sec_two_before = 0
        exp_two_before = 0
        spoke_two_before = 0
        gov_two_before = 0
        part_two_before = 0
        found_two_before = 0
        
        sec_word = ''

        sent_split = re.split(r'[ |,|.]', sent)
        last_word = re.split(r'[ ]', phrase_in_sent)[0]

        if last_word in sent_split:
            word_index = sent_split.index(last_word)
            sec_word = sent_split[word_index-2].lower()
            if word_index-2 >= 0:
                if sec_word == 'who':
                    who_two_before = 1;
                if sec_word == 'ceo':
                    ceo_two_before = 1;
                if sec_word == 'senator' or sec_word == 'sen':
                    sen_two_before = 1;
                if sec_word == 'president':
                    pres_two_before = 1;
                if sec_word == 'investor':
                    inv_two_before = 1;
                if sec_word == 'author':
                    aut_two_before = 1;
                if sec_word == 'representative' or sec_word == 'rep':
                    rep_two_before = 1;
                if sec_word == 'ambassador':
                    amb_two_before = 1;
                if sec_word == 'secretary':
                    sec_two_before = 1;
                if sec_word == 'expert':
                    exp_two_before = 1;
                if sec_word == 'spokesman' or sec_word == 'spokeswoman':
                    spoke_two_before = 1;
                if sec_word == 'governor':
                    gov_two_before = 1;
                if sec_word == 'partner':
                    part_two_before = 1;
                if sec_word == 'founder':
                    found_two_before = 1;
                return who_two_before,ceo_two_before,sen_two_before,pres_two_before,inv_two_before,aut_two_before,rep_two_before,amb_two_before,sec_two_before,exp_two_before,spoke_two_before,gov_two_before,part_two_before,found_two_before;
            else:
                return who_two_before,ceo_two_before,sen_two_before,pres_two_before,inv_two_before,aut_two_before,rep_two_before,amb_two_before,sec_two_before,exp_two_before,spoke_two_before,gov_two_before,part_two_before,found_two_before;
    except IndexError:  
        return who_two_before,ceo_two_before,sen_two_before,pres_two_before,inv_two_before,aut_two_before,rep_two_before,amb_two_before,sec_two_before,exp_two_before,spoke_two_before,gov_two_before,part_two_before,found_two_before;


# In[30]:


def person_one_before(sent,phrase_in_sent):
    try:
        who_one_before = 0
        ceo_one_before = 0
        sen_one_before = 0
        pres_one_before = 0
        inv_one_before = 0
        aut_one_before = 0
        rep_one_before = 0
        amb_one_before = 0
        sec_one_before = 0
        exp_one_before = 0
        spoke_one_before = 0
        gov_one_before = 0
        part_one_before = 0
        found_one_before = 0
        
        sec_word = ''

        sent_split = re.split(r'[ |,|.]', sent)
        last_word = re.split(r'[ ]', phrase_in_sent)[0]

        if last_word in sent_split:
            word_index = sent_split.index(last_word)
            sec_word = sent_split[word_index - 1].lower()
            if word_index - 1 >= 0:
                if sec_word == 'who':
                    who_one_before = 1;
                if sec_word == 'ceo':
                    ceo_one_before = 1;
                if sec_word == 'senator' or sec_word == 'sen':
                    sen_one_before = 1;
                if sec_word == 'president':
                    pres_one_before = 1;
                if sec_word == 'investor':
                    inv_one_before = 1;
                if sec_word == 'author':
                    aut_one_before = 1;
                if sec_word == 'representative' or sec_word == 'rep':
                    rep_one_before = 1;
                if sec_word == 'ambassador':
                    amb_one_before = 1;
                if sec_word == 'secretary':
                    sec_one_before = 1;
                if sec_word == 'expert':
                    exp_one_before = 1;
                if sec_word == 'spokesman' or sec_word == 'spokeswoman':
                    spoke_one_before = 1;
                if sec_word == 'governor':
                    gov_one_before = 1;
                if sec_word == 'partner':
                    part_one_before = 1;
                if sec_word == 'founder':
                    found_one_before = 1;
                return who_one_before,ceo_one_before,sen_one_before,pres_one_before,inv_one_before,aut_one_before,rep_one_before,amb_one_before,sec_one_before,exp_one_before,spoke_one_before,gov_one_before,part_one_before,found_one_before;
            else:
                return who_one_before,ceo_one_before,sen_one_before,pres_one_before,inv_one_before,aut_one_before,rep_one_before,amb_one_before,sec_one_before,exp_one_before,spoke_one_before,gov_one_before,part_one_before,found_one_before;
    except IndexError:  
        return who_one_before,ceo_one_before,sen_one_before,pres_one_before,inv_one_before,aut_one_before,rep_one_before,amb_one_before,sec_one_before,exp_one_before,spoke_one_before,gov_one_before,part_one_before,found_one_before;


# In[31]:


def person_one_after(sent,phrase_in_sent):
    try:
        who_one_after = 0
        ceo_one_after = 0
        sen_one_after = 0
        pres_one_after = 0
        inv_one_after = 0
        aut_one_after = 0
        rep_one_after = 0
        amb_one_after = 0
        sec_one_after = 0
        exp_one_after = 0
        spoke_one_after = 0
        gov_one_after = 0
        part_one_after = 0
        found_one_after = 0
        
        fst_word = ''

        sent_split = re.split(r'[ |,|.]', sent)
        last_word = re.split(r'[ ]', phrase_in_sent)[1]

        if last_word in sent_split:
            word_index = sent_split.index(last_word)
            fst_word = sent_split[word_index+1].lower()

            if fst_word == 'who':
                who_one_after = 1;
            if fst_word == 'ceo':
                ceo_one_after = 1;
            if fst_word == 'senator' or fst_word == 'sen':
                sen_one_after = 1;
            if fst_word == 'president':
                pres_one_after = 1;
            if fst_word == 'investor':
                inv_one_after = 1;
            if fst_word == 'author':
                aut_one_after = 1;
            if fst_word == 'representative'or fst_word == 'rep':
                rep_one_after = 1;
            if fst_word == 'ambassador':
                amb_one_after = 1;
            if fst_word == 'secretary':
                sec_one_after = 1;
            if fst_word == 'expert':
                exp_one_after = 1;
            if fst_word == 'spokesman' or fst_word == 'spokeswoman':
                spoke_one_after = 1;
            if fst_word == 'governor':
                gov_one_after = 1;
            if fst_word == 'partner':
                part_one_after = 1;
            if fst_word == 'founder':
                found_one_after = 1;
        return who_one_after,ceo_one_after,sen_one_after,pres_one_after,inv_one_after,aut_one_after,rep_one_after,amb_one_after,sec_one_after,exp_one_after,spoke_one_after,gov_one_after,part_one_after,found_one_after;
    except IndexError:  
        return who_one_after,ceo_one_after,sen_one_after,pres_one_after,inv_one_after,aut_one_after,rep_one_after,amb_one_after,sec_one_after,exp_one_after,spoke_one_after,gov_one_after,part_one_after,found_one_after;


# In[32]:


def person_two_after(sent,phrase_in_sent):
    try:
        who_two_after = 0
        ceo_two_after = 0
        sen_two_after = 0
        pres_two_after = 0
        inv_two_after = 0
        aut_two_after = 0
        rep_two_after = 0
        amb_two_after = 0
        sec_two_after = 0
        exp_two_after = 0
        spoke_two_after = 0
        gov_two_after = 0
        part_two_after = 0
        found_two_after = 0
        
        sec_word = ''

        sent_split = re.split(r'[ |,|.]', sent)
        last_word = re.split(r'[ ]', phrase_in_sent)[1]

        if last_word in sent_split:
            word_index = sent_split.index(last_word)
            sec_word = sent_split[word_index+2].lower()

            if sec_word == 'who':
                who_two_after = 1;
            if sec_word == 'ceo':
                ceo_two_after = 1;
            if sec_word == 'senator' or sec_word == 'sen':
                sen_two_after = 1;
            if sec_word == 'president':
                pres_two_after = 1;
            if sec_word == 'investor':
                inv_two_after = 1;
            if sec_word == 'author':
                aut_two_after = 1;
            if sec_word == 'representative'or sec_word == 'rep':
                rep_two_after = 1;
            if sec_word == 'ambassador':
                amb_two_after = 1;
            if sec_word == 'secretary':
                sec_two_after = 1;
            if sec_word == 'expert':
                exp_two_after = 1;
            if sec_word == 'spokesman' or sec_word == 'spokeswoman':
                spoke_two_after = 1;
            if sec_word == 'governor':
                gov_two_after = 1;
            if sec_word == 'partner':
                part_two_after = 1;
            if sec_word == 'founder':
                found_two_after = 1;
        return who_two_after,ceo_two_after,sen_two_after,pres_two_after,inv_two_after,aut_two_after,rep_two_after,amb_two_after,sec_two_after,exp_two_after,spoke_two_after,gov_two_after,part_two_after,found_two_after;
    except IndexError:  
        return who_two_after,ceo_two_after,sen_two_after,pres_two_after,inv_two_after,aut_two_after,rep_two_after,amb_two_after,sec_two_after,exp_two_after,spoke_two_after,gov_two_after,part_two_after,found_two_after;    


# In[33]:


def ceo_word_within_two(sent,phrase):
    try:
        two_before = person_two_before(sent,phrase)
        one_before = person_one_before(sent,phrase)
        one_after = person_one_after(sent,phrase)
        two_after = person_two_after(sent,phrase)

        who = two_before[0] + one_before[0] + one_after[0] + two_after[0]
        ceo_in_sent = two_before[1] + one_before[1] + one_after[1] + two_after[1]
        senator = two_before[2] + one_before[2] + one_after[2] + two_after[2]
        president = two_before[3] + one_before[3] + one_after[3] + two_after[3]
        investor = two_before[4] + one_before[4] + one_after[4] + two_after[4]
        author = two_before[5] + one_before[5] + one_after[5] + two_after[5]
        rep = two_before[6] + one_before[6] + one_after[6] + two_after[6]
        ambassador = two_before[7] + one_before[7] + one_after[7] + two_after[7]
        secretary = two_before[8] + one_before[8] + one_after[8] + two_after[8]
        expert = two_before[9] + one_before[9] + one_after[9] + two_after[9]
        spokesman = two_before[10] + one_before[10] + one_after[10] + two_after[10]
        governor = two_before[11] + one_before[11] + one_after[11] + two_after[11]
        partner = two_before[12] + one_before[12] + one_after[12] + two_after[12]
        founder = two_before[13] + one_before[13] + one_after[13] + two_after[13]

        return who,ceo_in_sent,senator, president, investor, author, rep, ambassador, secretary, expert, spokesman, governor, partner,founder;
    except TypeError:
        return np.zeros(14)


# In[34]:


def potential_ceo_df(ls):
    ceo_df = []
    sentences = []
    for i in range(len(ls)):
        p = re.findall(r'[A-Z]\w+ [A-Z]\w+', ls[i])
        if p != []:
            
            sent_caps = cap_in_sent(ls[i])
            sent_len = len(ls[i])
            
            for j in p:
                ceo_word = ceo_word_within_two(ls[i],j)
                who = ceo_word[0]
                ceos = ceo_word[1]
                sen_two = ceo_word[2]
                pres_two = ceo_word[3]
                inv_two = ceo_word[4]
                aut_two = ceo_word[5]
                rep_two = ceo_word[6]
                amb_two = ceo_word[7]
                sec_two = ceo_word[8]
                exp_two = ceo_word[9]
                spoke_two = ceo_word[10]
                gov_two = ceo_word[11]
                part_two = ceo_word[12]
                found_two = ceo_word[13]
                
                in_sent = sentence_words(ls[i])
                ceo_in_sent = in_sent[0]
                sens = in_sent[1]
                pres = in_sent[2]
                inv = in_sent[3]
                aut = in_sent[4]
                represent = in_sent[5]
                ambass = in_sent[6]
                secr = in_sent[7]
                exp = in_sent[8]
                spok = in_sent[9]
                gov = in_sent[10]
                part = in_sent[11]
                found = in_sent[12]
                who_in_sent = in_sent[13]
                            
                length = len(j)
                caps = cap_letters(j)
                ceo_df.append([j,length,sent_len,caps,sent_caps,who,ceos,sen_two,pres_two,inv_two,aut_two,rep_two,amb_two,sec_two,exp_two,spoke_two,gov_two,part_two,found_two,ceo_in_sent,sens,pres,inv,aut,represent,ambass,secr,exp,spok,gov,part,found,who_in_sent,ls[i],i])
                
    return ceo_df;


# In[35]:


ceo_df = pd.DataFrame(potential_ceo_df(sentences), columns = ['Candidate','length','sent_len','caps','sent_caps','who','ceos_two','sen_two','pres_two','inv_two','aut_two','rep_two','amb_two','sec_two','exp_two','spoke_two','gov_two','part_two','found_two','ceo_in_sent','sens','pres','inv','aut','represent','ambass','secr','exp','spok','gov','part','found','who_in_sent','Sentence','index'])


# In[36]:


ceo_df


# ## CEO Logistic Regression

# In[37]:


labels = []
values = ceo['ceo_full'].values

for i in range(len(ceo_df)):
    if ceo_df.loc[i,'Candidate'] in values:
        labels.append(1)
    else: 
        labels.append(0) 
ceo_df['label'] = labels


# In[38]:


ceo_df_final = ceo_df.drop(['Sentence','index','Candidate'], axis=1)


# In[39]:


ceo_df_final.sum(axis=0)


# In[40]:


yceo = ceo_df_final.loc[:, ceo_df_final.columns == 'label']
Xceo = ceo_df_final.loc[:, ceo_df_final.columns != 'label']


# ### Two Model Types:
#     1) Over-Sampling model
#         a. With all features from RFE
#         b. With select features from RFE
#     2) Regular model

# ### Model 1a: Over-Sampling (All Features)

# In[41]:


os = SMOTE(random_state=0)
Xceo_train, Xceo_test, yceo_train, yceo_test = train_test_split(Xceo, yceo, test_size=0.5, random_state=0)
columns = Xceo_train.columns

os_ceo_X,os_ceo_y=os.fit_sample(Xceo_train, yceo_train)
os_ceo_X = pd.DataFrame(data=os_ceo_X,columns=columns )
os_ceo_y= pd.DataFrame(data=os_ceo_y,columns=['label'])


# In[42]:


print("length of oversampled ceos is ",len(os_ceo_X))
print("Number of non-CEOs in oversampled ceos",len(os_ceo_y[os_ceo_y['label']==0]))
print("Number of CEOs",len(os_ceo_y[os_ceo_y['label']==1]))
print("Proportion of non-ceos in oversampled ceos is ",len(os_ceo_y[os_ceo_y['label']==0])/len(os_ceo_X))
print("Proportion of ceos in oversampled ceos is ",len(os_ceo_y[os_ceo_y['label']==1])/len(os_ceo_X))


# In[43]:


logceo = LogisticRegression()
rfe = RFE(logceo)
rfe = rfe.fit(os_ceo_X, os_ceo_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# In[44]:


predictors=['caps','ceos_two','sen_two','pres_two','inv_two','aut_two','rep_two','sec_two','exp_two','spoke_two','found_two','ceo_in_sent','represent','ambass','spok','found'] 
X=os_ceo_X[predictors]
y=os_ceo_y['label']


# In[45]:


logit_model=sm.Logit(y,X)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[46]:


predictors=['caps','ceos_two','pres_two','inv_two','sec_two','found_two','ceo_in_sent','represent','spok','found'] 
Xceo_os=os_ceo_X[predictors]
yceo_os=os_ceo_y['label']


# In[47]:


logit_model=sm.Logit(yceo_os,Xceo_os)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[48]:


Xceo_os_train, Xceo_os_test, yceo_os_train, yceo_os_test = train_test_split(Xceo_os, yceo_os, test_size=0.5, random_state=0)
log_ceo0S = LogisticRegression()
log_ceo0S.fit(Xceo_os_train, yceo_os_train)


# In[49]:


yceoOS_pred = log_ceo0S.predict(Xceo_os_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_ceo0S.score(Xceo_os_test, yceo_os_test)))


# In[50]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yceo_os_test.tolist(), yceoOS_pred.tolist())
print(confusion_matrix)


# In[51]:


print(classification_report(yceo_os_test, yceoOS_pred))


# ### Model 1b: Over-Sampling (Select Features)

# In[52]:


predictors1b=['ceos_two','pres_two','inv_two','sec_two','found_two'] 
Xceo1b_os=os_ceo_X[predictors1b]
yceo_os=os_ceo_y['label']


# In[53]:


logit_model=sm.Logit(yceo_os,Xceo1b_os)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[54]:


Xceo1b_os_train, Xceo1b_os_test, yceo_os_train, yceo_os_test = train_test_split(Xceo1b_os, yceo_os, test_size=0.5, random_state=0)
log_1bceo0S = LogisticRegression()
log_1bceo0S.fit(Xceo1b_os_train, yceo_os_train)


# In[55]:


yceoOS_pred1b = log_1bceo0S.predict(Xceo1b_os_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_1bceo0S.score(Xceo1b_os_test, yceo_os_test)))


# In[56]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yceo_os_test.tolist(), yceoOS_pred1b.tolist())
print(confusion_matrix)


# In[57]:


print(classification_report(yceo_os_test, yceoOS_pred1b))


# ### Model 2: Non-OS

# In[58]:


Xceo_train, Xceo_test, yceo_train, yceo_test = train_test_split(Xceo, yceo, test_size=0.5, random_state=0)


# In[59]:


ceo_log = LogisticRegression()
ceo_log.fit(Xceo_train[predictors], yceo_train)


# In[60]:


ceo_log.coef_


# In[61]:


ceo_pred = ceo_log.predict(Xceo_test[predictors])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(ceo_log.score(Xceo_test[predictors], yceo_test)))


# In[62]:


sum(ceo_pred)


# In[63]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yceo_test['label'].tolist(), ceo_pred.tolist())
print(confusion_matrix)


# In[64]:


print(classification_report(yceo_test, ceo_pred))


# ### Testing of Models on Entire Dataset:

# ### Model 1a

# In[65]:


yfinal_pred = log_ceo0S.predict(Xceo[predictors])
print('Accuracy of logistic regression classifier on entire data: {:.2f}'.format(log_ceo0S.score(Xceo[predictors], yceo)))


# In[66]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(yceo['label'].tolist(), yfinal_pred.tolist())
print(confusion_matrix)


# In[67]:


print(classification_report(yceo, yfinal_pred))


# ### Model 1b

# In[68]:


yfinal_pred1b = log_1bceo0S.predict(Xceo[predictors1b])
print('Accuracy of logistic regression classifier on entire data: {:.2f}'.format(log_1bceo0S.score(Xceo[predictors1b], yceo)))


# In[69]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(yceo['label'].tolist(), yfinal_pred1b.tolist())
print(confusion_matrix)


# In[70]:


print(classification_report(yceo, yfinal_pred1b))


# ### Model 2

# In[71]:


ceo_pred_full = ceo_log.predict(Xceo[predictors])
sum(ceo_pred_full)


# In[72]:


print('Accuracy of logistic regression classifier on entire data: {:.2f}'.format(ceo_log.score(Xceo[predictors], yceo)))


# In[73]:


print(classification_report(yceo, ceo_pred_full))


# In[74]:


from sklearn.metrics import confusion_matrix
cmf = confusion_matrix(yceo['label'].tolist(), ceo_pred_full.tolist())
print(cmf)


# ### Model blank is best by inspection

# In[75]:


# Model 1a
ceo_df['pred'] = yfinal_pred
ceo_final = ceo_df[ceo_df['pred']==1]
ceo_final = ceo_final.reset_index(drop=True)
CEOs = list(ceo_final['Candidate'])
set(CEOs)


# In[76]:


# Model 1b
ceo_df['pred'] = yfinal_pred1b
ceo_final = ceo_df[ceo_df['pred']==1]
ceo_final = ceo_final.reset_index(drop=True)
CEOs = list(ceo_final['Candidate'])
set(CEOs)


# In[163]:


# Model 2
ceo_df['pred'] = ceo_pred_full
ceo_final = ceo_df[ceo_df['pred']==1]
ceo_final = ceo_final.reset_index(drop=True)
CEOs = list(ceo_final['Candidate'])
set(CEOs)


# ### Model 2 preferred because of precision

# In[78]:


finalCEO = set(CEOs)
finalCEO = pd.DataFrame(finalCEO)
finalCEO.to_csv("ExtractedCEOs.csv",header=False,index=False)


# ## Companies

# In[79]:


def company_in_sentence(sentence):
    ret = 0
    if re.search(r'company', sentence.lower()) != None:
        ret = 1
    return ret


# In[80]:


def stock_in_sentence(sentence):
    ret = 0
    if re.search(r'stock', sentence.lower()) != None:
        ret = 1
    return ret


# In[81]:


def shares_in_sentence(sentence):
    ret = 0
    if re.search(r'share', sentence.lower()) != None:
        ret = 1
    return ret


# In[82]:


def trade_in_sentence(sentence):
    ret = 0
    if re.search(r'trad', sentence.lower()) != None:
        ret = 1
    return ret


# ### Company Specific 

# In[83]:


def length_of_company(item):
    return len(item)


# In[84]:


def plural_word(item):
    plural = 0
    if item[len(item) - 1] == 's':
        plural = 1
    return plural


# In[85]:


def number_of_words(words):
    return len(words)


# In[86]:


def location_at_start(sentence, item):
    start = 0
    if re.search(re.compile(item), sentence).start() == 0:
        start = 1;
    else:
        start = 0;
    return start;


# In[87]:


def company_words(word_phrase):
    corp = 0
    corporation = 0
    group = 0
    holding = 0
    inc = 0
    company = 0
    association = 0
    foundation = 0

    for word in word_phrase:
        if word == "Corp" or word == 'Corp.' or word == 'Corporation':
            corp = 1;
        if word == "Group":
            group = 1;
        if word == "Holding":
            holding = 1;
        if word == "Inc" or word == "Inc.":
            inc = 1;
        if word == "Company":
            company = 1;
        if word == "Association":
            association = 1;
        if word == "Foundation":
            foundation = 1;

    return corp, group, holding, inc, company, association, foundation


# In[88]:


def feature_creator_companies(sentences):
    candidates = []
    for i in range(len(sentences)):
        x = re.findall(r'(([A-Z][A-Za-z0-9]+[ -]?)+)', sentences[i])
        extract = [i[0] for i in x]
        if extract != []:
            comp_in_sent = company_in_sentence(sentences[i])
            stock = stock_in_sentence(sentences[i])
            shares = shares_in_sentence(sentences[i])
            trade = trade_in_sentence(sentences[i]) 
            for j in extract:
                
                new_j = j
                if new_j[-1] == ' ':
                    new_len = len(new_j)-1
                    new_j = new_j[0:new_len]
                
                words = re.split(r'[ ]', new_j)
                length = length_of_company(new_j)
                plural = plural_word(new_j)
                number_words = number_of_words(words)
                location = location_at_start(sentences[i], new_j)
                comp = company_words(words)
                corp = comp[0]
                group = comp[1]
                holding = comp[2]
                inc = comp[3]
                company = comp[4]
                association = comp[5]
                foundation = comp[6]
                candidates.append([new_j,comp_in_sent,stock,shares,trade,length,plural,number_words,location,corp,group,holding,inc,company,association,foundation,sentences[i],i])
    return candidates


# In[89]:


comp_df = pd.DataFrame(feature_creator_companies(sentences), columns = ['Candidate','comp_in_sent','stock','shares','trade','length','plural', 'number_words','location' , 'corp', 'group', 'holding', 'inc', 'company', 'association','foundation','sentence','index'])


# In[90]:


comp_df


# ### Logistic Regression for Companies

# In[91]:


comp_labels = []
values = set(company['company'].values)
candidates = comp_df['Candidate'].tolist()

for i in range(len(comp_df)):
    if candidates[i] in values:
        comp_labels.append(1)
    else: 
        comp_labels.append(0)
comp_df['label'] = comp_labels


# In[92]:


comp_df_final = comp_df.drop(['sentence','index','Candidate'], axis=1)


# In[93]:


comp_df_final.sum(axis=0)


# In[94]:


ycomp = comp_df_final.loc[:, comp_df_final.columns == 'label']
Xcomp = comp_df_final.loc[:, comp_df_final.columns != 'label']


# In[95]:


Xcomp_train, Xcomp_test, ycomp_train, ycomp_test = train_test_split(Xcomp, ycomp, test_size=0.5, random_state=0)


# ### Three Models:
#     1) Features based on p-values
#     2) Features based on RFE
#     3) Over-sampling model using RFE samples

# ### Model 1: P-values

# In[96]:


logit_model=sm.Logit(ycomp_train,Xcomp_train)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# #### Choosing factors with p-values under 0.05

# In[97]:


predictors1=['comp_in_sent','stock','shares','length','plural','number_words','location','corp','group','inc'] 
Xcomp_train1=Xcomp_train[predictors1]


# In[98]:


logit_model=sm.Logit(ycomp_train,Xcomp_train1)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[99]:


log_comp1 = LogisticRegression()
log_comp1.fit(Xcomp_train1, ycomp_train)


# In[100]:


ycomp_pred1 = log_comp1.predict(Xcomp_test[predictors1])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_comp1.score(Xcomp_test[predictors1], ycomp_test)))


# In[101]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(ycomp_test.iloc[:,0].tolist(), ycomp_pred1.tolist())
print(confusion_matrix)


# In[102]:


print(classification_report(ycomp_test, ycomp_pred1))


# ### Model 2: RFE

# In[103]:


logcomp = LogisticRegression()
rfe = RFE(logcomp)
rfe = rfe.fit(Xcomp_train, ycomp_train)
print(rfe.support_)
print(rfe.ranking_)


# In[104]:


predictors2=['shares','corp','group','inc','company','association','foundation'] 
Xcomp_train2=Xcomp_train[predictors2]


# In[105]:


logit_model=sm.Logit(ycomp_train,Xcomp_train2)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[106]:


log_comp2 = LogisticRegression()
log_comp2.fit(Xcomp_train2, ycomp_train)


# In[107]:


ycomp_pred2 = log_comp2.predict(Xcomp_test[predictors2])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_comp2.score(Xcomp_test[predictors2], ycomp_test)))


# In[108]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(ycomp_test.iloc[:,0].tolist(), ycomp_pred2.tolist())
print(confusion_matrix)


# In[109]:


print(classification_report(ycomp_test, ycomp_pred2))


# ### Model 3: Over-Sampling

# In[110]:


os = SMOTE(random_state=0)
columns = Xcomp_train.columns

os_comp_X,os_comp_y=os.fit_sample(Xcomp_train, ycomp_train)
os_comp_X = pd.DataFrame(data=os_comp_X,columns=columns )
os_comp_y= pd.DataFrame(data=os_comp_y,columns=['label'])


# In[111]:


print("length of oversampled comps is ",len(os_comp_X))
print("Number of non-companies in oversampled comps",len(os_comp_y[os_comp_y['label']==0]))
print("Number of companies",len(os_comp_y[os_comp_y['label']==1]))
print("Proportion of non-companies in oversampled comps is ",len(os_comp_y[os_comp_y['label']==0])/len(os_comp_X))
print("Proportion of companies in oversampled comps is ",len(os_comp_y[os_comp_y['label']==1])/len(os_comp_X))


# In[112]:


rfe = rfe.fit(os_comp_X, os_comp_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# In[113]:


os_comp_X.columns


# In[114]:


predictors3=['corp','group','inc','company','association','foundation'] 
Xcomp_os=os_comp_X[predictors3]
ycomp_os=os_comp_y['label']


# In[115]:


logit_model=sm.Logit(ycomp_os,Xcomp_os)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[116]:


Xcomp_os_train, Xcomp_os_test, ycomp_os_train, ycomp_os_test = train_test_split(Xcomp_os, ycomp_os, test_size=0.5, random_state=0)
log_compOS = LogisticRegression()
log_compOS.fit(Xcomp_os_train, ycomp_os_train)


# In[117]:


ycomp_os_pred = log_compOS.predict(Xcomp_os_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_compOS.score(Xcomp_os_test, ycomp_os_test)))


# In[118]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(ycomp_os_test.tolist(), ycomp_os_pred.tolist())
print(confusion_matrix)


# In[119]:


print(classification_report(ycomp_os_test, ycomp_os_pred))


# ### Comparison of 3 models on full dataset:

# ### Model 1:

# In[120]:


comp1_pred = log_comp1.predict(Xcomp[predictors1])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_comp1.score(Xcomp[predictors1], ycomp)))


# In[121]:


sum(comp1_pred)


# In[122]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(ycomp.iloc[:,0].tolist(), comp1_pred.tolist())
print(confusion_matrix)


# In[123]:


print(classification_report(ycomp, comp1_pred))


# ### Model 2

# In[124]:


comp2_pred = log_comp2.predict(Xcomp[predictors2])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_comp2.score(Xcomp[predictors2], ycomp)))


# In[125]:


sum(comp2_pred)


# In[126]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(ycomp.iloc[:,0].tolist(), comp2_pred.tolist())
print(confusion_matrix)


# In[127]:


print(classification_report(ycomp, comp2_pred))


# ### Model 3

# In[128]:


comp3_pred = log_compOS.predict(Xcomp[predictors3])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_compOS.score(Xcomp[predictors3], ycomp)))


# In[129]:


sum(comp3_pred)


# In[130]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(ycomp.iloc[:,0].tolist(), comp3_pred.tolist())
print(confusion_matrix)


# In[131]:


print(classification_report(ycomp, comp3_pred))


# ### Either the second or third model is the best based on confusion matrix and precision/recall on full dataset

# In[132]:


predictors2


# In[133]:


comp_df['pred'] = comp2_pred
comp_final = comp_df[comp_df['pred']==1]
comp_final = comp_final.reset_index(drop=True)
comps2 = list(comp_final['Candidate'])
comp_final


# In[134]:


comp_df['pred'] = comp3_pred
comp_final = comp_df[comp_df['pred']==1]
comp_final = comp_final.reset_index(drop=True)
comps1 = list(comp_final['Candidate'])
set(comps1)
comp_final


# ### Based on inspection of extracted values, the third model presents a better list

# In[135]:


finalCompany = set(comps1)
finalCompany = pd.DataFrame(finalCompany)
finalCompany.to_csv("ExtractedCompanies.csv",header=False,index=False)


# ## Percentages

# In[136]:


def percent_after(sent,num):
    try:
        perc = 0
        nxt = ''        
        split = re.split(r'[ ]', sent)
        if num in split:
            num_index = split.index(num)
            nxt = split[num_index+1].lower()
            if nxt == 'percentage' or nxt == "percent":
                perc = 1;
                return perc;
        char_index = re.search(num, sent.lower()).start() + len(num)
        if sent[char_index] == '%' or sent[char_index+1] == '%':
            perc = 1;
            return perc;
        else: perc = 0;
            
    except IndexError:
        perc = 0;
    return perc;


# In[137]:


def greater_than_1800(num):
    try:
        year = 0
        num = int(num)
        
        if num > 1800: year = 1;
        else: year = 0;
    except ValueError: pass
    return year;


# In[138]:


def feature_creator_percent(ls):
    numbers = []
    for i in range(len(ls)):
        re1 = re.findall(r'\d*\.?\d+', ls[i])
        re2 = re.findall(r'one[\s|-]?hundred|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen', ls[i].lower())
        re3 = re.findall(r'((twenty|thirty|fourty|fifty|sixty|seventy|eighty|ninety)(\s|-)?(one|two|three|four|five|six|seven|eight|nine)?)', ls[i].lower())
        re3 = [i[0] for i in re3]
        extract = re1 + re2 + re3
        if extract != []:
            for item in extract:
                year = greater_than_1800(item)
                perc = percent_after(ls[i],item)
                numbers.append([item, year, perc,i])
    return numbers


# In[139]:


numbers = pd.DataFrame(feature_creator_percent(sentences), columns = ['numbers','year','perc','sentence'])


# In[140]:


numbers


# ## Logistic Regression for Percentages

# In[141]:


def add_percentages(sent,num):
    percentage = num
    nxt = ''
    split = re.split(r'[ ]', sent)
    try:
        char_index = re.search(num, sent.lower()).start() + len(num)
        if sent[char_index] == '%':
            percentage = num + '%'
            return percentage;
        if sent[char_index+1] == '%':
            percentage = num + '%'
            return percentage; 

        if num in split:
            num_index = split.index(num)
            nxt = split[num_index+1].lower()
            if nxt == 'percentage':
                percentage = num + ' ' + nxt
                return percentage;
            if nxt == 'percent':
                percentage = num + ' ' + nxt
                return percentage;
            else:
                return percentage;
        else:
            return percentage;
    except IndexError:
        return percentage;


# In[142]:


percentages = []
for i in range(len(numbers)):
    sent = sentences[numbers.iloc[i,3]]
    num = numbers.iloc[i,0]
    percentages.append(add_percentages(sent,num))
    
numbers['numbers'] = percentages


# In[143]:


labels=[]
candidates = numbers['numbers'].tolist()
values = percent['perc'].values

for i in range(len(candidates)):
    if candidates[i] in values:
        labels.append(1)
    else: 
        labels.append(0) 
numbers['label'] = labels


# In[144]:


numbers.head()


# In[145]:


perc_df = numbers.drop(['numbers','sentence'], axis=1)


# In[146]:


perc_df.sum(axis=0)


# In[147]:


yperc = perc_df.loc[:, perc_df.columns == 'label']
Xperc = perc_df.loc[:, perc_df.columns != 'label']


# In[148]:


Xperc_train, Xperc_test, yperc_train, yperc_test = train_test_split(Xperc, yperc, test_size=0.5, random_state=0)


# In[149]:


logit_model=sm.Logit(yperc_train,Xperc_train)
result=logit_model.fit(method='bfgs')
print(result.summary2())


# In[150]:


log_perc = LogisticRegression()
log_perc.fit(Xperc_train, yperc_train)


# In[151]:


log_perc.coef_


# In[152]:


yperc_pred = log_perc.predict(Xperc_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_perc.score(Xperc_test, yperc_test)))


# In[153]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yperc_test.iloc[:,0].tolist(), yperc_pred.tolist())
print(confusion_matrix)


# In[154]:


print(classification_report(yperc_test, yperc_pred))


# In[155]:


full_perc_pred = log_perc.predict(Xperc)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_perc.score(Xperc, yperc)))


# In[156]:


sum(full_perc_pred)


# In[157]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yperc.iloc[:,0].tolist(), full_perc_pred.tolist())
print(confusion_matrix)


# In[158]:


print(classification_report(yperc, full_perc_pred))


# In[159]:


numbers['pred'] = full_perc_pred
perc_df_extract = numbers[numbers['pred']==1]
perc_df_extract = perc_df_extract.reset_index(drop=True)


# In[164]:


percentages = perc_df_extract['numbers']


# In[165]:


finalPercentage = pd.DataFrame(percentages)
finalPercentage.to_csv("ExtractedPercantages.csv",header=False,index=False)


# In[ ]:




