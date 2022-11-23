#!/usr/bin/env python
# coding: utf-8

# # Support Vector machine (SVM)
# it is kind of Supervised learning
# first of all we import the required Libraries
# #### اول از همه کتابخانه های مد نظر را ایمپورت میکنیم

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()


# In[3]:


cancer


# In[4]:


type(cancer)


# #### we want to see only the keys of our data (ommiting the values to prettify)

# In[5]:


cancer.keys()


# #### as we saw our data is a Dictionary type
# #### the (data) from (cancer) is a list and each member of (data) also is a list ! :)
# 

# In[6]:


print(cancer['data'])


# In[7]:


print(type(cancer['data']))


# In[8]:


# cancer positive or negative
# تشخیص اینکه سرطان دارند یا نه
print(cancer['target'])


# In[9]:


print(cancer['frame'])


# In[10]:


print(cancer['target_names'])


# In[11]:


print(cancer['feature_names'])


# In[12]:


print(cancer['filename'])


# In[13]:


#describing the dataset
print(cancer['DESCR'])


# In[14]:


print(type(cancer['feature_names']))


# #### we want to make a data frame that the data equals to cancer['data'] and the name of the columns equals to cancer['feature_names']
# #### میخواهیم یک دیتا ست بسازیم که دیتا و نام ستون های آن برگرفته از دو مورد زیر باشد
# ##### cancer['data']
# ##### cancer['feature_names']

# In[15]:


df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.head(3)


# In[16]:


df_feat.info()


# #### fortunately we dont have any missing value
# خوشبختانه دیتای میس شده نداریم

# In[17]:


cancer['target_names']


# #### malignant means bad type cancer and is shown by 0 in target frame
# ##### سرطان بدخیم را با صفر نشان میدهیم
# #### benign means goodtype cancer and is shown by 1 in target frame
# ##### سرطان خوش خیم را با یک نمایش می دهیم

# In[18]:


cancer['target']


# #### we have one column and we name it as cancer
# #### :فقط یک ستون به عنوان تارگت داریم و نام آنرا کنسر یا سرطان میگذاریم

# In[19]:


df_target=pd.DataFrame(cancer['target'],columns=['cancer'])


# #### combining features and target in one data frame and name it as df
# #### axis=1 means we want to combine in acordance with columns (not rows)
# دیتا بیس خود را تشکیل میدهیم

# In[20]:


df=pd.concat([df_feat,df_target],axis=1)


# In[21]:


df.head(1)


# #### lets explore our data

# In[22]:


sns.scatterplot(x='mean concavity',y='mean texture',hue='cancer',data=df)


# In[23]:


sns.scatterplot(x='worst texture',y='mean texture',hue='cancer',data=df)


# In[24]:


sns.scatterplot(x='mean radius',y='mean perimeter',hue='cancer',data=df)


# شمارش تعداد افرادی که سرطان خوشخیم و بدخیم دارند- یک و صفر
# 

# In[25]:


#let count the number of 0 & 1 in cancer column
sns.countplot(x='cancer',data=df)


# ### lets go trough ML

# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=(
    train_test_split(df_feat,np.ravel(df_target),test_size=0.3,random_state=101))
from sklearn.svm import SVC


# چون داده خروجی ما از نوع صفر و یک است ، بنابراین در حالت دسته بندی با کلاسیفایر قرار داریم . به همین دلیل از
# 
# svc
# 
# استفاده میکنیم.

# In[27]:


model=SVC()
model.fit(x_train,y_train)


# ### lets evaluate our Model

# In[29]:


predictions=model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# ### also we like to have classification Report:
# #### our model accuracy is 0.92
# #### our model precision is 0.95

# In[30]:


print(classification_report(y_test,predictions))


# #### because we want to gain better results we go to improve our model parameters
# #### press (shift+tab ) to see the default inputs for Model

# In[ ]:


#prsess shift+tab to show the default parameters which can be changed(press + icon after shift+tab)
model()


# #### we choose these three prameters and let the rest of the parameters at their default value
# #### we want to find the best value of these parameters (where we get the best results)
# #### we will have 25 state (5*5)

# In[31]:


param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}


# ### Note:
# this method (finding the best parameters) which is called Grid search can be applied also in other models (not only in svm model)

# In[33]:


from sklearn.model_selection import GridSearchCV


# #### we need some input for GridSearchCV.
# #### (refit True) means after each cycle refit the model
# #### as verbose get bigger value, we will get more detail (obviously more time consumption)

# In[34]:


# now we have(grid) instead of (model). so instead of saying (model.fit) , we say(grid.fit)
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[35]:


grid.fit(x_train,y_train)


# In[36]:


grid_predictions=grid.predict(x_test)


# #### now let use confusion matrix again to monitor the precision and accuracy
# #### remember the previous values:
# [ 56 10]
# 
# [ 3 102]

# In[37]:


print(confusion_matrix(y_test,grid_predictions))


# #### we can see the better results(56 trues in model and 59 trues in grid( after Optimization))
# #### Trues means both TP & TN (True positives and true Negatives)

# In[38]:


print(classification_report(y_test,grid_predictions))


# #### also our accuracy increased from 0.92 to 0.94
# ####  to see the best parameters:

# In[39]:


grid.best_params_


# #### conclusion
# #### مرور پایانی درس
# ####  ما از یک دیتا ست درونی خود سایکیت لرن استفاده کردیم که مربوط به سرطان سینه و  به شکل دیکشنری بود.  ما دیتا ست مورد نظر خود را از دیتای موجود تشکیل دادیم واز مدل
# #### SVM= support vector machine
# #### برای دسته بندی دیتا استفاده کردیم. علاوه بر آن از گرید سرچ استفاده کرده  و گفتیم که گرید سرچ بر روی تمام مدل‌هایی که هایپرپارامتر دارند، قابل استفاده است.   
# 
# #### به عبارت ساده گرید سرچ مقادیر دیفالت پارامترهای یک مدل را به گونه ای تغییر می‌دهد که نتایج بهتری حاصل شود

# In[ ]:




