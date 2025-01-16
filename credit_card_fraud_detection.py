
#importing the data manipulation and data visualisation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

#reading the credit card dataset
#reading the credit card dataset
df_cc = pd.read_csv('/content/creditcard.csv', on_bad_lines='skip')
# Removed 'error_bad_lines=False' as it's deprecated.
df_cc.head()

#Displays the last 5 rows from the dataset
df_cc.tail()

#Displays the no of rows and columns
df_cc.shape

#Displays the column names
df_cc.columns

#Checks if there is any null values in the dataset
df_cc.isna().sum()

df_cc.dropna()


df_cc = df_cc.dropna()

df_cc.isna().sum()

#Checks if there is any duplicates in the dataset
df_cc.duplicated().sum()

#Deletes all the duplicates from the dataset
df_cc.drop_duplicates()

#Counts the value of the column -class
df_cc["Class"].value_counts()

#DATA VISUALISATION

ax=sns.countplot(x='Class',data=df_cc,width=0.3)
ax.set_title("Distribution of transaction")
ax.set_xlabel("Type of transaction")
ax.set_ylabel("Number of transaction")

for i in ax.containers:
  ax.bar_label(i)
  plt.show()

plt.figure(figsize=(20,12))
plt.figure(figsize=(20,12))
ax=sns.heatmap(df_cc.corr(),annot=True,fmt='.2f')
for i in range(df_cc.shape[1]+1):
  ax.axvline(i,color='white',lw=2)
  ax.axhline(i,color='white',lw=2)
plt.show()

#Create a function that seperates the class of transaction between legit and fraud

def split_data_by_class(df_cc):
  legit = df_cc[df_cc['Class']==0]
  fraud = df_cc[df_cc['Class']==1]
  return legit, fraud

# Call the function and assign the returned values to variables
# This line is outside of the function, so the variables are accessible globally
legit_df, fraud_df = split_data_by_class(df_cc)

#example usage
#legit_df,fraud_df=split_data_by_class(df_cc)
legit_df.info()

fraud_df.info()

legit_df.describe().T

fraud_df.describe().T

#Sampling the legal transactions to match 492 rows
new_legit_df=legit_df.sample(n=492)
print(new_legit_df)

#combine the fraud and legit datasets
combine_df=pd.concat([new_legit_df,fraud_df],axis=0)
combine_df

combine_df["Class"].value_counts()

X=combine_df.drop(columns="Class",axis=1)
Y=combine_df["Class"]

X

Y

#Importing the model building libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#splitting and training data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=5)
print(X_train)

print(X_train.shape,X_test.shape)

#Building model

model=LogisticRegression()
model.fit(X_train,Y_train)

#Training model

X_train_predict=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_predict,Y_train)
print(f"The models accuracy on traing data is: {round(training_data_accuracy*100,2)}%")

#Model predition

X_test_predict=model.predict(X_test)
training_data_accuracy=accuracy_score(X_test_predict,Y_test)
print(f"The models accuracy on traing data is: {round(training_data_accuracy*100,2)}%")

from sklearn.metrics import ConfusionMatrixDisplay
ax=ConfusionMatrixDisplay.from_predictions(Y_test,X_test_predict)
plt.show()
