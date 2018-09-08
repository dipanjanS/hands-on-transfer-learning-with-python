
# coding: utf-8

# # Feature Engineering
import numpy as np
import pandas as pd

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

sns.set_style('whitegrid')
sns.set_context('talk')


# ## Feature Engineering : Numerical Data 

# load dataset 
credit_df = pd.read_excel('credit_default.xls',
                             skiprows=1,index_col=0)
credit_df.shape


credit_df.head()


# ###  Extract Raw Features
# Attributes which are useful in their raw form itself

credit_df[['LIMIT_BAL','BILL_AMT1',
                   'BILL_AMT2','BILL_AMT3',
                   'BILL_AMT4','BILL_AMT5',
                   'BILL_AMT6']].head()


# ### Counts
# Based on requirements, count of events is also a useful attribute.

# utility function
def default_month_count(row):
    count = 0 
    for i in [0,2,3,4,5,6]:
        if row['PAY_'+str(i)] > 0:
            count +=1
    return count



credit_df['number_of_default_months'] = credit_df.apply(default_month_count,
                                                         axis=1)


credit_df[['number_of_default_months']].head()


# ### Binarization
# Occurance or absence of an event is also a useful feature

credit_df['has_ever_defaulted'] = credit_df.number_of_default_months.apply(lambda x: 1 if x>0 else 0)
credit_df[['number_of_default_months','has_ever_defaulted']].head()


# ### Binning
# Also known as quantization, helps in transformin continuous features such as 
# age and income onto discrete scales.

credit_df.AGE.plot(kind='hist',bins=60)
plt.title('Age Histogram', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.xlabel('Frequency', fontsize=12)


# #### Fixed Width Bins

# Fixed Width Bins :
# 
# ``` 
# Age Range: Bin
# ---------------
#  0 -  9  : 0
# 10 - 19  : 1
# 20 - 29  : 2
# 30 - 39  : 3
#   ... and so on
# ```

# Assign a bin label to each row
credit_df['age_bin_fixed'] = credit_df.AGE.apply(lambda age: np.floor(age/10.))


credit_df[['AGE','age_bin_fixed']].head()


# #### Quantile Based Binning
# * 4-Quartile Binning

## Quantile binning
quantile_list = [0, .25, .5, .75, 1.]
quantiles = credit_df.AGE.quantile(quantile_list)
quantiles


# Plot Quartile Ranges on the Distribution

fig, ax = plt.subplots()
credit_df.AGE.plot(kind='hist',bins=60)

for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)

ax.set_title('Age Histogram with Quantiles', fontsize=12)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)


# Assign Quartile Bin Labels
quantile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
credit_df['age_quantile_range'] = pd.qcut(credit_df['AGE'],
                                          q=quantile_list)
credit_df['age_quantile_label'] = pd.qcut(credit_df['AGE'],
                                          q=quantile_list,
                                          labels=quantile_labels)


credit_df[['AGE','age_quantile_range','age_quantile_label']].head()



# ## Feature Engineering : Categorical Data
# We have utilized multiple publicly available datasets to better understand 
# categorical attributes

battles_df = pd.read_csv('battles.csv')
battles_df.shape


battles_df[['name','year','attacker_king','attacker_1']].head()


# ### Transforming Nominal Features
# Categorical attributes which ***do not*** have any intrinsic 
# ordering amongst them 

from sklearn.preprocessing import LabelEncoder

attacker_le = LabelEncoder()
attacker_labels = attacker_le.fit_transform(battles_df.attacker_1)
attacker_mappings = {index: label for index, label in enumerate(attacker_le.classes_)}
attacker_mappings


# assign labels
battles_df['attacker1_label'] = attacker_labels
battles_df[['name','year','attacker_king','attacker_1','attacker1_label']].head()


# ### Transforming Ordinal Features
# Categorical attributes which ***have*** an intrinsic ordering amongst them

sales_df = pd.DataFrame(data={
                            'items_sold':abs(np.random.randn(7)*100),
                             'day_of_week':['Monday', 'Tuesday',
                                            'Wednesday', 'Thursday', 
                                            'Friday', 'Saturday', 
                                            'Sunday']})
sales_df

day_map = {'Monday': 1, 'Tuesday': 2, 
           'Wednesday': 3, 'Thursday': 4, 
           'Friday': 5, 'Saturday': 6, 
           'Sunday' : 7}

sales_df['weekday_label'] = sales_df['day_of_week'].map(day_map)
sales_df.head()


# ### Encoding Categoricals 

# One Hot Encoder

from sklearn.preprocessing import OneHotEncoder

day_le = LabelEncoder()
day_labels = day_le.fit_transform(sales_df['day_of_week'])
sales_df['label_encoder_day_label'] = day_labels

# encode day labels using one-hot encoding scheme
day_ohe = OneHotEncoder()
day_feature_arr = day_ohe.fit_transform(sales_df[['label_encoder_day_label']]).toarray()
day_feature_labels = list(day_le.classes_)
day_features = pd.DataFrame(day_feature_arr, columns=day_feature_labels)


sales_ohe_df = pd.concat([sales_df, day_features], axis=1)
sales_ohe_df


# Dummy Encoder


day_dummy_features = pd.get_dummies(sales_df['day_of_week'], drop_first=True)
pd.concat([sales_df[['day_of_week','items_sold']], day_dummy_features], axis=1)

