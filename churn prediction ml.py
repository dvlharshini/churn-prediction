#!/usr/bin/env python
# coding: utf-8

# In[76]:


get_ipython().system('pip install streamlit-option-menu')


# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,MinMaxScaler,PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,confusion_matrix 


# In[78]:


df=pd.read_csv(r"C:\Users\dvlha\OneDrive\Documents\churn prediction application[1]\churn prediction application\CHURN DATA SET.csv")


# In[79]:


df


# In[80]:


df["state"].astype(str).tolist()


# In[81]:


df["area_code"]=df["area_code"].apply(lambda x:x[-3:])


# In[82]:


df.describe()


# In[83]:


df.head()


# In[84]:


df.info()


# In[85]:


df["area_code"]=df["area_code"].astype("int")


# In[86]:


df[df.isnull()].sum()


# In[87]:


df[df.duplicated]


# In[88]:


#detecting the outliers with boxplot


# In[89]:


# df2 with only numeric vaiables

df2=df.drop(['state', 'churn', 'international_plan', 'voice_mail_plan',"number_customer_service_calls"],axis=1)




for i in df2.columns:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[i])
    plt.title(f'Boxplot of {i}')
    plt.show()




# In[90]:


# fuction for caping the outliers by iqr method

def iqr(column):
    q1 = np.quantile(column, 0.25)
    q3 = np.quantile(column, 0.75)
    rang = q3 - q1
    right = q3 + rang * 1.5
    left = q1 - rang * 1.5
    
    # Caping extreme values
    column[column > right] = right
    column[column < left] = left

    return column


# In[91]:


for i in df2:
    
    df[i] = iqr(df[i])




# In[ ]:






# In[ ]:





# In[92]:


for i in df2.columns:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[i])


# In[93]:


plt.figure(figsize=(30,10))
sns.barplot(x=df["state"].value_counts().index, y=df["state"].value_counts().values)
plt.ylabel("count")
plt.show()


# In[94]:


df["state"].unique()


# In[95]:


df["state"].nunique()


# In[96]:


sns.barplot(x=df["international_plan"].value_counts().index, y=df["international_plan"].value_counts().values)
plt.ylabel("count")
plt.show()


# In[97]:


df["international_plan"].unique()


# In[98]:


df["international_plan"].nunique()


# In[99]:


#voice mail plan


# In[100]:


sns.barplot(x=df["voice_mail_plan"].value_counts().index, y=df["voice_mail_plan"].value_counts().values)
plt.ylabel("count")
plt.show()


# In[101]:


df["voice_mail_plan"].unique()


# In[102]:


df["voice_mail_plan"].nunique()


# In[103]:


df.head()


# # account length

# In[104]:


df["account_length"].describe()


# In[105]:


df["account_length"].value_counts()


# In[106]:


df["account_length"].unique()


# In[107]:


df["account_length"].min()


# In[108]:


df["account_length"].max()


# In[109]:


sns.distplot(df["account_length"])
plt.show()


# In[110]:


df["account_length"].skew()


# # area code

# In[111]:


print("Description of 'area_code' column:")
print(df["area_code"].describe())

print(" ")
print("*************************************************************")
print(" ")



print("\nUnique values of 'area_code' column:")
print(df["area_code"].unique())

print(" ")
print("*************************************************************")
print(" ")

print("Minimum value of 'area_code' column:", df["area_code"].min())
print("Maximum value of 'area_code' column:", df["area_code"].max())

print(" ")
print("*************************************************************")
print(" ")


print("\nDistribution of 'area_code' column:")
sns.distplot(df["area_code"])
plt.show()

print(" ")
print("*************************************************************")
print(" ")


print("\nSkewness of 'area_code' column:", df["area_code"].skew())


# In[112]:


sns.barplot(x=df["area_code"].value_counts().index,y=df["area_code"].value_counts().values)


# In[113]:


print("Description of 'number_vmail_messages' column:")
print(df["number_vmail_messages"].describe())



print(" ")
print("*************************************************************")
print(" ")


print("\nUnique values of 'number_vmail_messages' column:")
print(df["number_vmail_messages"].unique())

print(" ")
print("*************************************************************")
print(" ")


print("Minimum value of 'number_vmail_messages' column:", df["number_vmail_messages"].min())
print("Maximum value of 'number_vmail_messages' column:", df["number_vmail_messages"].max())

print(" ")
print("*************************************************************")
print(" ")




# In[114]:


df["number_vmail_messages"].nunique()


# In[115]:


# Plot distribution of "number_vmail_messages" column
print("\nDistribution of 'number_vmail_messages' column:")
sns.distplot(df["number_vmail_messages"])
plt.show()

print(" ")
print("*************************************************************")
print(" ")



# Skewness of "number_vmail_messages" column
print("\nSkewness of 'number_vmail_messages' column:", df["number_vmail_messages"].skew())


# In[116]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(30, 10))
sns.barplot(x=df["number_vmail_messages"].value_counts().index, y=df["number_vmail_messages"].value_counts().values)

# Loop through each bar to add text annotations
for index, value in enumerate(df["number_vmail_messages"].value_counts().values):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')

plt.show()


# In[117]:


df


# In[118]:


# Describe the "total_day_minutes" column
print("Description of 'total_day_minutes' column:")
print(df["total_day_minutes"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_day_minutes' column
print("Unique values of 'total_day_minutes' column:")
print(df["total_day_minutes"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_day_minutes' column
print("Minimum value of 'total_day_minutes' column:", df["total_day_minutes"].min())
print("Maximum value of 'total_day_minutes' column:", df["total_day_minutes"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_day_minutes' column
print("Number of unique values in 'total_day_minutes' column:", df["total_day_minutes"].nunique())



# In[119]:


# Plot distribution of "total_day_minutes" column
print("\nDistribution of 'total_day_minutes' column:")
sns.distplot(df["total_day_minutes"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_day_minutes" column
print("Skewness of 'total_day_minutes' column:", df["total_day_minutes"].skew())


# In[120]:


import seaborn as sns
import matplotlib.pyplot as plt

# Describe the "total_day_calls" column
print("Description of 'total_day_calls' column:")
print(df["total_day_calls"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_day_calls' column
print("Unique values of 'total_day_calls' column:")
print(df["total_day_calls"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_day_calls' column
print("Minimum value of 'total_day_calls' column:", df["total_day_calls"].min())
print("Maximum value of 'total_day_calls' column:", df["total_day_calls"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_day_calls' column
print("Number of unique values in 'total_day_calls' column:", df["total_day_calls"].nunique())

# Plot count of occurrences of "total_day_calls" column
print("\nCount of occurrences of 'total_day_calls' column:")
plt.figure(figsize=(30,20))
sns.countplot(x=df["total_day_calls"])
plt.show()

print("\n*************************************************************\n")


# In[121]:


# Describe the "total_day_charge" column
print("Description of 'total_day_charge' column:")
print(df["total_day_charge"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_day_charge' column
print("Unique values of 'total_day_charge' column:")
print(df["total_day_charge"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_day_charge' column
print("Minimum value of 'total_day_charge' column:", df["total_day_charge"].min())
print("Maximum value of 'total_day_charge' column:", df["total_day_charge"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_day_charge' column
print("Number of unique values in 'total_day_charge' column:", df["total_day_charge"].nunique())



# In[122]:


# Plot distribution of "total_day_charge" column
print("\nDistribution of 'total_day_charge' column:")
sns.distplot(df["total_day_charge"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_day_charge" column
print("Skewness of 'total_day_charge' column:", df["total_day_charge"].skew())


# In[123]:


import seaborn as sns
import matplotlib.pyplot as plt

# Describe the "total_eve_minutes" column
print("Description of 'total_eve_minutes' column:")
print(df["total_eve_minutes"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_eve_minutes' column
print("Unique values of 'total_eve_minutes' column:")
print(df["total_eve_minutes"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_eve_minutes' column
print("Minimum value of 'total_eve_minutes' column:", df["total_eve_minutes"].min())
print("Maximum value of 'total_eve_minutes' column:", df["total_eve_minutes"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_eve_minutes' column
print("Number of unique values in 'total_eve_minutes' column:", df["total_eve_minutes"].nunique())



# In[124]:


# Plot distribution of "total_eve_minutes" column
print("\nDistribution of 'total_eve_minutes' column:")
sns.distplot(df["total_eve_minutes"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_eve_minutes" column
print("Skewness of 'total_eve_minutes' column:", df["total_eve_minutes"].skew())



# In[125]:


# Describe the "total_eve_calls" column
print("Description of 'total_eve_calls' column:")
print(df["total_eve_calls"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_eve_calls' column
print("Unique values of 'total_eve_calls' column:")
print(df["total_eve_calls"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_eve_calls' column
print("Minimum value of 'total_eve_calls' column:", df["total_eve_calls"].min())
print("Maximum value of 'total_eve_calls' column:", df["total_eve_calls"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_eve_calls' column
print("Number of unique values in 'total_eve_calls' column:", df["total_eve_calls"].nunique())

# Plot count of occurrences of "total_eve_calls" column
print("\nCount of occurrences of 'total_eve_calls' column:")
plt.figure(figsize=(30,20))
sns.countplot(x=df["total_eve_calls"])
plt.show()

print("\n*************************************************************\n")



# In[126]:


# Describe the "total_eve_charge" column
print("\nDescription of 'total_eve_charge' column:")
print(df["total_eve_charge"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_eve_charge' column
print("Unique values of 'total_eve_charge' column:")
print(df["total_eve_charge"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_eve_charge' column
print("Minimum value of 'total_eve_charge' column:", df["total_eve_charge"].min())
print("Maximum value of 'total_eve_charge' column:", df["total_eve_charge"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_eve_charge' column
print("Number of unique values in 'total_eve_charge' column:", df["total_eve_charge"].nunique())



# In[127]:


# Plot distribution of "total_eve_charge" column
print("\nDistribution of 'total_eve_charge' column:")
sns.distplot(df["total_eve_charge"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_eve_charge" column
print("Skewness of 'total_eve_charge' column:", df["total_eve_charge"].skew())


# In[128]:


import seaborn as sns
import matplotlib.pyplot as plt

# Describe the "total_night_minutes" column
print("Description of 'total_night_minutes' column:")
print(df["total_night_minutes"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_night_minutes' column
print("Unique values of 'total_night_minutes' column:")
print(df["total_night_minutes"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_night_minutes' column
print("Minimum value of 'total_night_minutes' column:", df["total_night_minutes"].min())
print("Maximum value of 'total_night_minutes' column:", df["total_night_minutes"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_night_minutes' column
print("Number of unique values in 'total_night_minutes' column:", df["total_night_minutes"].nunique())

# Plot distribution of "total_night_minutes" column
print("\nDistribution of 'total_night_minutes' column:")
sns.distplot(df["total_night_minutes"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_night_minutes" column
print("Skewness of 'total_night_minutes' column:", df["total_night_minutes"].skew())



# In[129]:


# Describe the "total_night_calls" column
print("Description of 'total_night_calls' column:")
print(df["total_night_calls"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_night_calls' column
print("Unique values of 'total_night_calls' column:")
print(df["total_night_calls"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_night_calls' column
print("Minimum value of 'total_night_calls' column:", df["total_night_calls"].min())
print("Maximum value of 'total_night_calls' column:", df["total_night_calls"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_night_calls' column
print("Number of unique values in 'total_night_calls' column:", df["total_night_calls"].nunique())

# Plot count of occurrences of "total_night_calls" column
print("\nCount of occurrences of 'total_night_calls' column:")
plt.figure(figsize=(30,20))
sns.countplot(x=df["total_night_calls"])
plt.show()

print("\n*************************************************************\n")



# In[130]:


df["total_night_calls"].value_counts()


# In[131]:


# Describe the "total_night_charge" column
print("\nDescription of 'total_night_charge' column:")
print(df["total_night_charge"].describe())

print("\n*************************************************************\n")


print("\n*************************************************************\n")

# Minimum and maximum values of 'total_night_charge' column
print("Minimum value of 'total_night_charge' column:", df["total_night_charge"].min())
print("Maximum value of 'total_night_charge' column:", df["total_night_charge"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_night_charge' column
print("Number of unique values in 'total_night_charge' column:", df["total_night_charge"].nunique())



# In[132]:


# Plot distribution of "total_night_charge" column
print("\nDistribution of 'total_night_charge' column:")
sns.distplot(df["total_night_charge"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_night_charge" column
print("Skewness of 'total_night_charge' column:", df["total_night_charge"].skew())



# In[133]:


# Describe the "total_intl_minutes" column
print("\nDescription of 'total_intl_minutes' column:")
print(df["total_intl_minutes"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_intl_minutes' column
print("Unique values of 'total_intl_minutes' column:")
print(df["total_intl_minutes"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_intl_minutes' column
print("Minimum value of 'total_intl_minutes' column:", df["total_intl_minutes"].min())
print("Maximum value of 'total_intl_minutes' column:", df["total_intl_minutes"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_intl_minutes' column
print("Number of unique values in 'total_intl_minutes' column:", df["total_intl_minutes"].nunique())



# In[134]:


# Plot distribution of "total_intl_minutes" column
print("\nDistribution of 'total_intl_minutes' column:")
sns.distplot(df["total_intl_minutes"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_intl_minutes" column
print("Skewness of 'total_intl_minutes' column:", df["total_intl_minutes"].skew())



# In[135]:


# Describe the "total_intl_calls" column
print("Description of 'total_intl_calls' column:")
print(df["total_intl_calls"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_intl_calls' column
print("Unique values of 'total_intl_calls' column:")
print(df["total_intl_calls"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_intl_calls' column
print("Minimum value of 'total_intl_calls' column:", df["total_intl_calls"].min())
print("Maximum value of 'total_intl_calls' column:", df["total_intl_calls"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_intl_calls' column
print("Number of unique values in 'total_intl_calls' column:", df["total_intl_calls"].nunique())

# Plot count of occurrences of "total_intl_calls" column
print("\nCount of occurrences of 'total_intl_calls' column:")
sns.countplot(x=df["total_intl_calls"])
plt.show()

print("\n*************************************************************\n")



# In[136]:


# Describe the "total_intl_charge" column
print("\nDescription of 'total_intl_charge' column:")
print(df["total_intl_charge"].describe())

print("\n*************************************************************\n")

# Unique values of 'total_intl_charge' column
print("Unique values of 'total_intl_charge' column:")
print(df["total_intl_charge"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'total_intl_charge' column
print("Minimum value of 'total_intl_charge' column:", df["total_intl_charge"].min())
print("Maximum value of 'total_intl_charge' column:", df["total_intl_charge"].max())

print("\n*************************************************************\n")

# Number of unique values in 'total_intl_charge' column
print("Number of unique values in 'total_intl_charge' column:", df["total_intl_charge"].nunique())



# In[ ]:





# In[137]:


# Plot distribution of "total_intl_charge" column
print("\nDistribution of 'total_intl_charge' column:")
sns.distplot(df["total_intl_charge"])
plt.show()

print("\n*************************************************************\n")

# Skewness of "total_intl_charge" column
print("Skewness of 'total_intl_charge' column:", df["total_intl_charge"].skew())


# In[138]:


# Describe the "number_customer_service_calls" column
print("Description of 'number_customer_service_calls' column:")
print(df["number_customer_service_calls"].describe())

print("\n*************************************************************\n")

# Unique values of 'number_customer_service_calls' column
print("Unique values of 'number_customer_service_calls' column:")
print(df["number_customer_service_calls"].unique())

print("\n*************************************************************\n")

# Minimum and maximum values of 'number_customer_service_calls' column
print("Minimum value of 'number_customer_service_calls' column:", df["number_customer_service_calls"].min())
print("Maximum value of 'number_customer_service_calls' column:", df["number_customer_service_calls"].max())

print("\n*************************************************************\n")

# Number of unique values in 'number_customer_service_calls' column
print("Number of unique values in 'number_customer_service_calls' column:", df["number_customer_service_calls"].nunique())

# Plot count of occurrences of "number_customer_service_calls" column
print("\nCount of occurrences of 'number_customer_service_calls' column:")
sns.countplot(x=df["number_customer_service_calls"],hue=df["churn"])
plt.show()

print("\n*************************************************************\n")


# In[139]:


df.columns


# In[140]:


df.head()


# In[141]:


columns_for_pairplot = ['account_length', 'total_day_minutes', 'total_day_calls', 'total_day_charge','total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge']


sns.pairplot(df[columns_for_pairplot])
plt.show()


# In[142]:


df.columns


# In[143]:


df2 = df.drop(['state', 'churn', 'international_plan', 'voice_mail_plan'], axis=1)

correlation_matrix = df2.corr()

# Plot heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f")
plt.title('Correlation Matrix')
plt.show()


# # churn vs state

# In[144]:


plt.figure(figsize=(15, 15))
sns.countplot(x="state", hue="churn", data=df)
plt.title("churn vs state")
plt.show()


# In[145]:


plt.figure(figsize=(8,4))
sns.countplot(x="international_plan", hue="churn", data=df)
plt.title("international_plan vs churn")
plt.show()


# In[146]:


a=(len(df[(df["international_plan"]=="yes") & (df["churn"]=="yes")])/len(df[(df["international_plan"]=="yes")]))*100
b=(len(df[(df["international_plan"]=="no") & (df["churn"]=="yes")])/len(df[(df["international_plan"]=="no")]))*100



print(a, "are churn out of all international planed counstomers")
print(b, "are churn out of all non international planed counstomers")


# # area code vs state

# In[147]:


plt.figure(figsize=(8,4))
sns.countplot(x="area_code", hue="churn", data=df)
plt.title("area_code vs churn")
plt.show()


# # voice_mail_plan vs churn

# In[148]:


plt.figure(figsize=(8,4))
sns.countplot(x="voice_mail_plan", hue="churn", data=df)
plt.title("voice_mail_plan vs churn")
plt.show()


# In[149]:


df


# # number_customer_service_calls	 vs churn

# In[150]:


plt.figure(figsize=(10,10))
sns.countplot(x="number_customer_service_calls", hue="churn", data=df)
plt.title("number_customer_service_calls vs churn")
plt.show()


# # feature creation

# In[151]:


df.columns


# In[152]:


len(df)


# In[153]:


l_total_min=[df["total_day_minutes"][i]+df['total_eve_minutes'][i]+df['total_night_minutes'][i] for i in range(len(df))]


# In[154]:


l_total_call=[df["total_day_calls"][i]+df['total_eve_calls'][i]+df['total_night_calls'][i] for i in range(len(df))]


# In[155]:


l_total_charge=[df["total_day_charge"][i]+df['total_eve_charge'][i]+df['total_night_charge'][i] for i in range(len(df))]


# In[156]:


df["total_min"]=l_total_min
df["total_call"]=l_total_call
df["total_charge"]=l_total_charge


# In[157]:


l_days=[i*30 for i in df["account_length"]]


# In[158]:


l_weeks=[i/4 for i in df["account_length"]]


# In[159]:


l_year=[i/12 for i in df["account_length"]]


# In[160]:


df["plan_day"]=l_days


# In[161]:


df["plan_weeks"]=l_weeks


# In[162]:


df["plan_years"]=l_year


# In[163]:


l_charge_day=df["total_charge"]/df["plan_day"]


# In[164]:


df["charge_day"]=l_charge_day


# In[165]:


df


# In[166]:


df.columns


# # model building
# # Split data into training and testing sets

# In[167]:


x = df.drop('churn', axis=1)
y = df['churn']


# In[168]:


x.columns


# In[169]:


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[170]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)


# In[171]:


#pipeline of decision tree


# In[172]:


import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[173]:


ohe_cols = ['state', 'international_plan', 'voice_mail_plan']

scale_cols = ['account_length', 'area_code', 'number_vmail_messages', 'total_day_minutes', 
              'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 
              'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 
              'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 
              'number_customer_service_calls', 'total_min', 'total_call', 'total_charge', 
              'plan_day', 'plan_weeks', 'plan_years', 'charge_day']


# In[174]:


pipeline_dec = Pipeline([
    ('data_preparation', ColumnTransformer([('onehot', OneHotEncoder(), ohe_cols),('scaling', MinMaxScaler(), scale_cols)])),
    ('classifier', DecisionTreeClassifier(splitter="best", criterion="gini", max_depth=5, min_samples_split=2, min_samples_leaf=5))
])



# In[175]:


pipeline_dec.fit(x_train,y_train)


# # evalution metrix

# In[176]:


y_test_pred=pipeline_dec.predict(x_test)
y_train_pred=pipeline_dec.predict(x_train)


# In[177]:


test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred)


# In[178]:


train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)


# In[179]:


print("Training Evaluation Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1 Score:", train_f1)
print("ROC AUC Score:", train_roc_auc)





# In[180]:


print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print("Test ROC AUC Score:", test_roc_auc)


# In[181]:


import matplotlib.pyplot as plt
import numpy as np

# Data for training and test evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
training_metrics = [0.972156862745098, 0.975, 0.832, 0.897841726618705, 0.9141609195402298]
test_metrics = [0.9752941176470589, 0.9689119170984456, 0.8385650224215246, 0.8990384615384616, 0.9172513669995234]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_metrics, width, label='Training')
rects2 = ax.bar(x + width/2, test_metrics, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Training vs Test Evaluation Metrics for the Decision Tree')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# # cross val score

# In[182]:


from sklearn.model_selection import cross_validate


cv_results = cross_validate(pipeline_dec, x_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])




# In[183]:


print("\nTesting Scores:")
print("Accuracy:", cv_results['test_accuracy'].mean())
print("Precision:", cv_results['test_precision'].mean())
print("Recall:", cv_results['test_recall'].mean())
print("F1 Score:", cv_results['test_f1'].mean())
print("ROC AUC Score:", cv_results['test_roc_auc'].mean())


# # random forest pipeline

# In[184]:


pipeline_random_forest = Pipeline([
    ('data_preparation', ColumnTransformer([('onehot', OneHotEncoder(), ohe_cols),('scaling', MinMaxScaler(), scale_cols)])),
    ('classifier', RandomForestClassifier(
        n_estimators=70, 
        criterion="gini",
        max_depth=13,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42 
    ))
])



# In[185]:


pipeline_random_forest.fit(x_train,y_train)


# In[186]:


y_test_pred=pipeline_random_forest.predict(x_test)
y_train_pred=pipeline_random_forest.predict(x_train)


# In[187]:


test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred)


# In[188]:


train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)


# In[189]:


print("Training Evaluation Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1 Score:", train_f1)
print("ROC AUC Score:", train_roc_auc)





# In[190]:


print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print("Test ROC AUC Score:", test_roc_auc)


# In[191]:


import matplotlib.pyplot as plt
import numpy as np

# Data for training and test evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
training_metrics = [0.9788235294117648, 1.0, 0.856, 0.9224137931034483, 0.928]
test_metrics = [0.961764705882353, 0.9817073170731707, 0.7219730941704036, 0.8320413436692508, 0.8599709749795824]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_metrics, width, label='Training')
rects2 = ax.bar(x + width/2, test_metrics, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Training vs Test Evaluation Metrics of Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[192]:


from sklearn.model_selection import cross_validate


cv_results = cross_validate(pipeline_random_forest, x_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])




# In[193]:


print("\nTesting Scores:")
print("Accuracy:", cv_results['test_accuracy'].mean())
print("Precision:", cv_results['test_precision'].mean())
print("Recall:", cv_results['test_recall'].mean())
print("F1 Score:", cv_results['test_f1'].mean())
print("ROC AUC Score:", cv_results['test_roc_auc'].mean())


# # pipeline for the knn

# # Kneighbor classifier

# In[194]:


pipeline_knn = Pipeline([
    ('data_preparation', ColumnTransformer([('onehot', OneHotEncoder(), ohe_cols),('scaling', MinMaxScaler(), scale_cols)])),
    ('classifier',KNeighborsClassifier(n_neighbors=10) )
])



# In[195]:


pipeline_knn.fit(x_train,y_train)


# In[196]:


y_test_pred=pipeline_knn.predict(x_test)
y_train_pred=pipeline_knn.predict(x_train)


# In[197]:


test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred)


# In[198]:


train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)


# In[199]:


print("Training Evaluation Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1 Score:", train_f1)
print("ROC AUC Score:", train_roc_auc)





# In[200]:


print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1 Score:", test_f1)
print("Test ROC AUC Score:", test_roc_auc)


# In[201]:


import matplotlib.pyplot as plt
import numpy as np

# Data for training and test evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
training_metrics = [0.8678431372549019, 0.8958333333333334, 0.11466666666666667, 0.20330969267139481, 0.556183908045977]
test_metrics = [0.8747058823529412, 0.8125, 0.05829596412556054, 0.1087866108786611, 0.5281324099571607]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_metrics, width, label='Training')
rects2 = ax.bar(x + width/2, test_metrics, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Training vs Test Evaluation Metrics of KNN')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[202]:


cv_results = cross_validate(pipeline_knn, x_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])




# In[203]:


print("\nTesting Scores:")
print("Accuracy:", cv_results['test_accuracy'].mean())
print("Precision:", cv_results['test_precision'].mean())
print("Recall:", cv_results['test_recall'].mean())
print("F1 Score:", cv_results['test_f1'].mean())
print("ROC AUC Score:", cv_results['test_roc_auc'].mean())


# In[204]:


# svc


# In[205]:


pipeline_svm = Pipeline([
    ('data_preparation', ColumnTransformer([('onehot', OneHotEncoder(), ohe_cols),('scaling', MinMaxScaler(), scale_cols)])),
   ('classifier', SVC(kernel="sigmoid", C=1))
])



# In[206]:


pipeline_svm.fit(x_train,y_train)


# In[207]:


y_test_pred=pipeline_svm.predict(x_test)
y_train_pred=pipeline_svm.predict(x_train)


# In[208]:


test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred)


# In[209]:


train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)


# In[210]:


train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)


# In[211]:


print("Training Evaluation Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1 Score:", train_f1)
print("ROC AUC Score:", train_roc_auc)





# In[212]:


import matplotlib.pyplot as plt
import numpy as np

# Data for training and test evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
training_metrics = [
    0.8419607843137255, 
    0.25, 
    0.037333333333333336, 
    0.06496519721577727, 
    0.5090114942528736
]
test_metrics = [
    0.8652941176470588, 
    0.4117647058823529, 
    0.06278026905829596, 
    0.10894941634241245, 
    0.524619653825018
]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_metrics, width, label='Training')
rects2 = ax.bar(x + width/2, test_metrics, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Training vs Test Evaluation Metrics of SVC')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[213]:


cv_results = cross_validate(pipeline_svm, x_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])




# In[214]:


print("\nTesting Scores:")
print("Accuracy:", cv_results['test_accuracy'].mean())
print("Precision:", cv_results['test_precision'].mean())
print("Recall:", cv_results['test_recall'].mean())
print("F1 Score:", cv_results['test_f1'].mean())
print("ROC AUC Score:", cv_results['test_roc_auc'].mean())


# # # by the cross val score we finalise the decision tree model

# In[215]:


# deploying the model


# In[216]:


pip install streamlit


# In[217]:


from sklearn.pipeline import Pipeline


# In[ ]:





# In[218]:


import pickle
import streamlit as st
# Save the pipeline
with open("strnew1.pkl", "wb") as f:
    pickle.dump(Pipeline, f)


# In[219]:


predict=pickle.load(open("strnew1.pkl","rb"))


# In[ ]:





# In[220]:


predict.predict(x_train)


# In[ ]:


input=x_train.iloc[:1,:]


# In[ ]:


input


# In[ ]:


predict.predict(input)


# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Define the models and their respective evaluation metrics
models = ['Decision Tree', 'Random Forest', 'KNN', 'SVC']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']

# Scores for each model
scores = {
    'Decision Tree': [0.9678431372549021, 0.9580279955595643, 0.8186666666666668, 0.8812063067003717, 0.9177624521072797],
    'Random Forest': [0.943529411764706, 0.9793567209848429, 0.6293333333333333, 0.7658206482488022, 0.913704214559387],
    'KNN': [0.8619607843137256, 0.8880952380952382, 0.07200000000000001, 0.13287531335822061, 0.7268045977011494],
    'SVC': [0.8454901960784313, 0.3094871794871795, 0.04266666666666667, 0.07430479338277116, 0.522752490421456]
}

# Convert scores to a NumPy array for easier manipulation
score_array = np.array([scores[model] for model in models])

# Plotting
x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plot each model's scores as a set of bars
for i, model in enumerate(models):
    ax.bar(x + i*width - width*1.5, score_array[i], width, label=model)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Cross-Validation Scores by Model and Metric')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Attach a text label above each bar in rects, displaying its height.
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call autolabel function for each set of bars
for i, model in enumerate(models):
    autolabel(ax.containers[i])

fig.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




