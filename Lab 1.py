#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit


# In[22]:


df = pd.read_csv('wine.csv')


# In[23]:


df.head()


# In[24]:


sns.boxplot(x=df['quality'], y=df['chlorides']);


# In[25]:


q1, q3 = np.percentile(df,[25, 75])
print(q1, q3)


# In[26]:


iqr = q3-q1
print(iqr)

1.5*iqr
# In[27]:


df.describe()


# In[28]:


df.info()


# In[29]:


df.quality.value_counts()


# In[30]:


df.describe()


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


df.hist(bins = 50, figsize = (20,15))
plt.show();


# In[33]:


print("Median alcohol range", df["alcohol"].min(), df["alcohol"].max())
print("ph median range", df.pH.min(), df.pH.max())
print("Median quality range", df.quality.min(), df.quality.max())


# In[34]:


def split_train_test(df,test_ratio,random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(test_ratio * len(df))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices],df.iloc[test_indices]

train_set,test_set = split_train_test(df,0.2)
print("Train set length:", len(train_set))
print("Test set length:", len(test_set))


# In[35]:


df['alcohol'].hist();


# In[36]:


df['quality'] = pd.cut(df['alcohol'],bins=[0,4.5,9.5,10.5,11.5,12.5,  np.inf], labels=[3,4,5,6,7,8])


# In[37]:


df.quality.hist();


# In[38]:


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['quality']):
    print(train_index)
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[39]:


for set_ in (strat_train_set,strat_test_set):
    set_.drop("quality",axis=1,inplace=True)


# In[40]:


df = strat_train_set.copy()


# In[41]:


#диаграмма рассеяния
df.plot(kind='scatter',x='alcohol',y = 'pH',alpha = 0.1);


# In[42]:


df.plot(kind='scatter',x='alcohol',y = 'density',alpha = 0.1,s=df['fixed acidity']/100,label='fixed acidity',figsize=(10,7),c='volatile acidity',cmap = plt.get_cmap('jet'),colorbar=True)
plt.legend()


# In[43]:


df


# In[44]:


corr_matrix = df.corr()
corr_matrix['pH'].sort_values(ascending=False)


# In[45]:


from pandas.plotting import scatter_matrix

attributes = ['pH', 'volatile acidity', 'alcohol', 'free sulfur dioxide']
scatter_matrix(df[attributes],figsize=(10,7));


# In[46]:


df.plot(kind='scatter',x='volatile acidity',y='pH',alpha=0.1)


# In[47]:


df


# In[48]:


corr_matrix = df.corr()
corr_matrix['pH'].sort_values(ascending=False)


# In[49]:


needed_features = ['volatile_acidity', 'alcohol', 'free sulfur dioxide', 'total sulfur oxide', 'sulphates', 'chlorides','citric acid', 'fixed acidity']


# In[50]:


df = strat_train_set.drop('pH',axis=1)
housing_labels = strat_train_set['pH'].copy()


# In[51]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')


# In[52]:


df_num = df.drop('residual sugar',axis = 1)
imputer.fit(df_num)
imputer.statistics_ == df_num.median().values


# In[53]:


X = imputer.transform(df_num)


# In[54]:


df_tranformed = pd.DataFrame(X, columns=df_num.columns,index=df_num.index)
df_tranformed


# In[55]:


df.dropna(subset=['chlorides'])
df.drop('chlorides',axis = 1)
median = df['chlorides'].median()
df.chlorides.fillna(median,inplace=True)


# In[56]:


df_category = df[['residual sugar']]
df_category


# In[57]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()


# In[58]:


category_encoded = ordinal_encoder.fit_transform(df_category)
category_encoded[:10]


# In[59]:


ordinal_encoder.categories_


# In[60]:


from sklearn.preprocessing import OneHotEncoder
onehotEnconder = OneHotEncoder()
df_category_one_hot = onehotEnconder.fit_transform(df_category)
df_category_one_hot.toarray()


# In[61]:


from sklearn.base import BaseEstimator, TransformerMixin
alcohol_ix,sulphates_ix,chlorides_ix,density_idx = 3,4,5,6

class CombinedAttributeAdder(BaseEstimator,TransformerMixin):
    def __init__(self,sulphates_per_alcohol = False):
        self.sulphates_per_alcohol =sulphates_per_alcohol

    def fit(self,X,y = None):
        return self

    def transform(self,X,y=None):
        alcohol_per_density = X[:, alcohol_ix] / X[:, density_idx]
        chlorides_per_density = X[:, chlorides_ix] / X[:, density_idx]
        if self.sulphates_per_alcohol:
            sulphates_per_alcohol = X[:, sulphates_ix] / X[:, alcohol_ix]
            return np.c_[X,alcohol_per_density,chlorides_per_density,sulphates_per_alcohol]
        else:
            return np.c_[X, alcohol_per_density, chlorides_per_density]


# In[62]:


additive_transformer = CombinedAttributeAdder(sulphates_per_alcohol = False)
df_extra_attributes = additive_transformer.transform(df.values)


# In[63]:


df_extra_attributes


# In[64]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[65]:


pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('attribs_adder',CombinedAttributeAdder()),
    ('std_scaler', StandardScaler())
])

df_num_transformed = pipeline.fit_transform(df_num)


# In[66]:


df_num_transformed


# In[67]:


from sklearn.compose import ColumnTransformer
num_attributes = list(df_num)
cat_attributes = ['chlorides']

full_pipeline = ColumnTransformer([
    ("num", pipeline,num_attributes),
    ("cat", OneHotEncoder(),cat_attributes)
])


df_prepared = full_pipeline.fit_transform(df)


# In[68]:


df_extra_attributes


# In[69]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[72]:


pipeline = Pipeline([
    ('imputer' , SimpleImputer(strategy = 'median')),
    ('attribs_adder', CombinedAttributeAdder()),
    ('std_scaler', StandardScaler())
])

df_num_transformed = pipeline.fit_transform(df_num)


# In[73]:


df_num_transformed


# In[77]:


from sklearn.compose import ColumnTransformer
num_attributes = list(df_num)
cat_attributes = ['chlorides']

full_pipeline = ColumnTransformer([
    ("num", pipeline,num_attributes),
    ("cat", OneHotEncoder(),cat_attributes)
])


df_prepared = full_pipeline.fit_transform(df)


# In[78]:


class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        return X[self.attribute_names].values


# In[79]:


old_num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributeAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])


# In[88]:


from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])


# In[92]:


old_df_prepared = old_full_pipeline.fit_transform(df)

