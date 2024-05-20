#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("adult.csv")
df.head(5)


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.describe


# In[6]:


df = df.replace('?', np.NaN)
df.head


# In[7]:


df.isna().sum()


# In[8]:


df["workclass"].mode()[0]


# In[9]:


df["occupation"].mode()[0]


# In[10]:


df["native.country"].mode()[0]


# In[11]:


df["workclass"].fillna(df["workclass"].mode()[0], inplace=True)
df["occupation"].fillna(df["occupation"].mode()[0], inplace=True)
df["native.country"].fillna(df["native.country"].mode()[0], inplace=True)

df["workclass"].describe()


# In[12]:


df.describe


# In[13]:


df.isna().sum()


# In[14]:


sns.heatmap(df.corr(),cmap = 'Greens',annot = True)


# In[15]:


df.hist(figsize = (10,10), color = "Green")


# In[16]:


df.plot(kind = 'box', figsize = (12,12), layout = (3,3), sharex = False, subplots = True);


# In[18]:


df['income'].value_counts()


# In[19]:


sns.countplot(df['income'], palette='coolwarm', hue='sex', data=df);


# In[20]:



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

num_col = ['age', 'fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
df.num = df[num_col]
df.num


# # PCA

# In[21]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit_transform(df.num)


# In[22]:


print(pca.components_)
print(sum(pca.explained_variance_ratio_))


# In[23]:


pca.explained_variance_ratio_


# In[24]:


X= df.drop(['income'], axis=1)
y = df['income']


# In[25]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[26]:


df2= df.copy()
df2= df2.apply(LabelEncoder().fit_transform)
df2.head()


# In[27]:


ss= StandardScaler().fit(df2.drop('income', axis=1))


# In[28]:


X= ss.transform(df2.drop('income', axis=1))
y= df['income']


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# # Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

model = lr.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Accuracy on training data: {:,.3f}".format(lr.score(X_train, y_train)))
print("Accuracy on test data: {:,.3f}".format(lr.score(X_test, y_test)))


# # Random Forest Classifier

# In[34]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

model1 = rfc.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

print("Accuracy on training data: {:,.3f}".format(rfc.score(X_train, y_train)))
print("Accuracy on test data: {:,.3f}".format(rfc.score(X_test, y_test)))

