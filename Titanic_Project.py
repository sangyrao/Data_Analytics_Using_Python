
# coding: utf-8

# 
#    
#    1.Who are the passengers on the Titatnic ? (Age ,Gender,class,..etc)
#    2.What are the passengers on and how does that relate to their class ?
#    3.Where did the passengers come from ?
#    4.Who was alone and who was with family ?
#    5.What factors helped someone survive the sinking?
# 

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[39]:


titanic_df=pd.read_csv('train.csv'); #reads a csv file containing data.


# In[40]:


titanic_df.head() #shows the few rows of the dataframe titanic_df


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().magic(u'matplotlib inline')


# In[41]:


sns.factorplot('Sex',data=titanic_df,kind='count')  #shows the visulisation of gender in the titanic_df dataframe.


# In[42]:


sns.factorplot('Sex',data=titanic_df,hue='Pclass',kind='count') # hue gives the small table(chart) guide for specific data in graph


# In[43]:


sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count') #shows Pclassin row and distinguishes gender.


# In[9]:


#function created by user to calculate the number of male , female , and childs in the  database. 
def male_female_child(passenger):
    age,sex=passenger
    if age<16:
      return 'child'
    else:
      return sex 


# In[47]:


titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1) 
#creates a new column person by combining age and sex . applies the function that we created .
                   


# In[11]:


titanic_df[0:10]


# In[12]:


sns.factorplot('Pclass',data=titanic_df,hue='person',kind='count')  


# In[13]:


titanic_df['Age'].hist(bins=70)  #visualises histogram of age


# In[14]:


titanic_df['Age'].mean()  # takes the mean  of age of persons in titanic


# In[15]:


titanic_df['person'].value_counts()  #counts number of male , female,and childs.


# In[16]:


fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4) # creates Facetgrid graph of Sex
fig.map(sns.kdeplot,'Age',shade=True) # graph of age

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()


# In[17]:


fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()


# In[18]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))

fig.add_legend()


# In[19]:


titanic_df.head()


# In[20]:


deck = titanic_df['Cabin'].dropna() #drops the Cabin column and stores only nonnull value to deck.


# In[21]:


deck.head()


# In[22]:


levels = []

for level in deck:
    levels.append(level[0]) #shows different levels.
    
cabin_df = DataFrame(levels)
cabin_df.columns=['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count')


# In[23]:


cabin_df=cabin_df[cabin_df.Cabin !='T'] #removes T column
sns.factorplot('Cabin',data=cabin_df,palette='summer',kind='count')


# In[24]:


titanic_df.head()


# In[25]:


sns.factorplot('Embarked',data=titanic_df,hue='Pclass',kind='count')


# In[26]:


#who was alone and who was with family ?
titanic_df.head()


# In[49]:


titanic_df['Alone']=titanic_df.SibSp + titanic_df.Parch


# In[50]:


titanic_df['Alone'] #if zero then they had some family with them otherwise 1.


# In[51]:


titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[52]:


titanic_df.head()


# In[53]:


sns.factorplot('Alone',data=titanic_df,palette='Blues',kind='count')


# In[54]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')


# In[55]:


sns.factorplot('Pclass','Survived',data=titanic_df)


# In[56]:


sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)


# In[57]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[58]:


sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')


# In[59]:


generations=[10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[60]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
