
# coding: utf-8

# 
#    
#    1.Who are the passengers on the Titatnic ? (Age ,Gender,class,..etc)
#    2.What are the passengers on and how does that relate to their class ?
#    3.Where did the passengers come from ?
#    4.Who was alone and who was with family ?
#    5.What factors helped someone survive the sinking?
# 

# In[45]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[46]:


titanic_df=pd.read_csv('train.csv');


# In[47]:


titanic_df.head()


# In[48]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


get_ipython().magic(u'matplotlib inline')


# In[50]:


sns.factorplot('Sex',data=titanic_df,kind='count')


# In[51]:


sns.factorplot('Sex',data=titanic_df,hue='Pclass',kind='count')


# In[52]:


sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count')


# In[53]:


def male_female_child(passenger):
    age,sex=passenger
    if age<16:
      return 'child'
    else:
      return sex 


# In[54]:


titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[55]:


titanic_df[0:10]


# In[57]:


sns.factorplot('Pclass',data=titanic_df,hue='person',kind='count')


# In[58]:


titanic_df['Age'].hist(bins=70)


# In[59]:


titanic_df['Age'].mean()


# In[60]:


titanic_df['person'].value_counts()


# In[70]:


fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()


# In[71]:


fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()


# In[72]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()


# In[73]:


titanic_df.head()


# In[76]:


deck = titanic_df['Cabin'].dropna()


# In[77]:


deck.head()


# In[80]:


levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns=['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count')


# In[84]:


cabin_df=cabin_df[cabin_df.Cabin !='T']
sns.factorplot('Cabin',data=cabin_df,palette='summer',kind='count')


# In[85]:


titanic_df.head()


# In[97]:


sns.factorplot('Embarked',data=titanic_df,hue='Pclass',kind='count')


# In[98]:


#who was alone and who was with family ?
titanic_df.head()


# In[100]:


titanic_df['Alone']=titanic_df.SibSp + titanic_df.Parch


# In[103]:


titanic_df['Alone'] #if zero then they had some familie with them otherwise 1.


# In[108]:


titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[109]:


titanic_df.head()


# In[110]:


sns.factorplot('Alone',data=titanic_df,palette='Blues',kind='count')


# In[115]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')


# In[116]:


sns.factorplot('Pclass','Survived',data=titanic_df)


# In[117]:


sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)


# In[121]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[122]:


sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')


# In[127]:


generations=[10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[131]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)

