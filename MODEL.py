#!/usr/bin/env python
# coding: utf-8

# ### Importing Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[2]:


cancer = pd.read_csv('cancer dataset.csv')


# In[3]:


cancer.head()


# In[4]:


cancer.shape


# ### Data Preprocessing

# In[5]:


cancer.isnull().sum()


# In[6]:


cancer.describe()


# In[7]:


cancer.drop('Unnamed: 32', axis=1, inplace=True)


# In[8]:


cancer.head()


# In[9]:


pd.get_dummies(cancer['diagnosis'], drop_first = True)


# In[10]:


target = pd.get_dummies(cancer['diagnosis'], drop_first=True)


# In[11]:


cancer = pd.concat([cancer, target], axis = 1)


# In[12]:


cancer.drop(['diagnosis', 'id'], inplace = True, axis = 1)


# In[13]:


cancer.head()


# ### Exploratery Data Analysis

# In[14]:


sns.set_style("whitegrid")


# In[15]:


sns.countplot(data=cancer, x='M', palette= 'hls')


# In[16]:


sns.boxplot(x= "M",y= "radius_mean", data= cancer)


# In[17]:


cancer["radius_mean"].plot.hist(figsize= (10,5))


# In[18]:


cancer["area_mean"].plot.hist(figsize= (10,5),color= "pink")


# ### Model Building - Logistic Regression

# In[19]:


X = cancer.drop('M', axis=1)
Y = cancer['M']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.3, random_state= 1)


# In[22]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
  
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 


# In[23]:


from sklearn.decomposition import PCA 
  
pca = PCA(n_components = 15) 
  
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
  
explained_variance = pca.explained_variance_ratio_ 


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


reg = LogisticRegression(random_state= 0 )


# In[26]:


#X_train = pca.fit_transform(X_train) 
#X_test = pca.transform(X_test) 


# In[27]:


reg.fit(X_train, Y_train)


# In[28]:


y_pred = reg.predict(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix


# In[30]:


confusion_matrix(Y_test, y_pred)


# In[31]:





# In[32]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))


# In[33]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(reg,X_test,Y_test)
plt.show()


# In[34]:


#Saving model to disk

pickle.dump(reg, open('model.pkl', 'wb'))


# In[35]:


model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:




