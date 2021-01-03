#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')


# In[60]:


messages=pd.read_csv('/Users/chocz/Documents/spam.csv',encoding='latin-1')


# In[61]:


ps=PorterStemmer()


# In[62]:


corpus=[]


# In[63]:


messages[['message','label']]=messages[['v2','v1']]


# In[64]:


for i in range(0,len(messages)):
    lines=re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    lines=lines.lower()
    lines=lines.split()
    lines=[ps.stem(word) for word in lines if not word in stopwords.words('english')]
    lines=' '.join(lines)
    corpus.append(lines)


# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()


# In[66]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# In[67]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[68]:


from sklearn.naive_bayes import MultinomialNB


# In[69]:


spam_detect_model=MultinomialNB().fit(x_train,y_train)


# In[70]:


y_pred=spam_detect_model.predict(x_test)


# In[71]:


y_pred


# In[ ]:




