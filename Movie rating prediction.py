#importing the nessacary header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df_movie=pd.read_csv('/content/IMDb Movies India.csv',engine='python',encoding='latin-1')
df_movie.head()

df_movie.shape

df_movie.describe()

df_movie.isna().sum()

df_movie.dropna(inplace=True)
df_movie.isna().sum()

#Data preprocessing

# Replace 'nan' with empty string and convert to numeric, coercing errors to NaN
df_movie['Year'] = pd.to_numeric(df_movie['Year'].astype(str).str.replace(r'[()]','',regex=True).str.replace('nan', ''), errors='coerce')
# Now you can safely convert to int, filling NaN values with a suitable integer like 0 or -1
df_movie['Year'] = df_movie['Year'].fillna(0).astype(int)
# Convert 'Duration' column to string type before applying string methods
df_movie['Duration'] = df_movie['Duration'].astype(str)
#Now apply string methods and convert it to numeric
# Remove 'min' and any surrounding whitespace from 'Duration' and convert to numeric
df_movie['Duration'] = pd.to_numeric(df_movie['Duration'].str.replace(r'\s*min\s*', '', regex=True), errors='coerce')
df_movie['Genre'] = df_movie['Genre'].str.split(',')
df_movie = df_movie.explode('Genre')
df_movie['Genre'].fillna(df_movie['Genre'].mode()[0], inplace=True)

# Convert 'Votes' column to string type before applying string methods
df_movie['Votes'] = df_movie['Votes'].astype(str)  # This line is added to convert 'Votes' to string

# Now, apply the string methods
df_movie['Votes'] = pd.to_numeric(df_movie['Votes'].str.replace(r'[$,M]', '', regex=True), errors='coerce')

df_movie.info()
df_movie.info()

#Data visualisation

year=px.histogram(df_movie,x='Year',histnorm='probability density',nbins=30)
year.show()

avg_rating=df_movie.groupby(['Year','Genre'])['Rating'].mean().reset_index()
top_genres=df_movie['Genre'].value_counts().head(10).index
avg_rating=avg_rating[avg_rating['Genre'].isin(top_genres)]
fig=px.line(avg_rating,x='Year',y='Rating',color='Genre',title='Average Rating Over Time by Genre')
fig.update_layout(xaxis_title='year',yaxis_title='average rating')
fig.show()

rating_fig=px.histogram(df_movie,x='Rating',nbins=40,title='distribution of Ratings',histnorm='probability density')
rating_fig.update_layout(xaxis_title='Rating',yaxis_title='Frequency',title_x=0.5,title_pad=dict(t=20),title_font=dict(size=20))
rating_fig.show()

#Feature engineering

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,mean_absolute_error

if 'Name' in df_movie.columns:
    df_movie.drop('Name', axis=1, inplace=True)
genre_mean_rating=df_movie.groupby('Genre')['Rating'].transform('mean')
df_movie['Genre_Mean_Rating']=genre_mean_rating
director_mean_rating=df_movie.groupby('Director')['Rating'].transform('mean')
df_movie['Director_encoded']=director_mean_rating
actor1_mean_rating=df_movie.groupby('Actor 1')['Rating'].transform('mean')
df_movie['Actor1_encoded']=actor1_mean_rating
actor2_mean_rating=df_movie.groupby('Actor 2')['Rating'].transform('mean')
df_movie['Actor2_encoded']=actor2_mean_rating
actor3_mean_rating=df_movie.groupby('Actor 3')['Rating'].transform('mean')
df_movie['Actor3_encoded']=actor3_mean_rating

x=df_movie[['Year','Duration','Genre_Mean_Rating','Director_encoded','Actor1_encoded','Actor2_encoded','Actor3_encoded']]
y=df_movie['Rating']
x = pd.DataFrame(x)
y = pd.DataFrame(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



# Now you can use dropna on x_train and y_train
x_train.dropna(inplace=True)

#Building and training a model

# Create separate imputers for x and y
imputer_x = SimpleImputer(strategy='mean')  # For x_train and x_test
imputer_y = SimpleImputer(strategy='mean')  # For y_train

# Fit and transform x_train
x_train = imputer_x.fit_transform(x_train)

# Fit and transform y_train
y_train = imputer_y.fit_transform(y_train.values.reshape(-1, 1))
y_train = y_train.ravel()  # Flatten back to original shape

# Now you can use dropna on x_train and y_train, ensuring consistency
x_train = pd.DataFrame(x_train) # Convert x_train back to DataFrame
# Reset index to avoid potential misalignment issues
x_train.reset_index(drop=True, inplace=True)
y_train = pd.Series(y_train).reset_index(drop=True) # Reset index for y_train

# Apply dropna after imputation
x_train.dropna(inplace=True)
y_train = y_train[x_train.index] # Align y_train with x_train after dropna

# Impute missing values in x_test using the same imputer for x
x_test = imputer_x.transform(x_test) # Use transform, not fit_transform, for x_test

model = LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("The performance evaluation of Logistic regresion is below:",'\n')
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))

#Model testing

x.head(5)

y.head(5)

data={'Year':[2019],'Duration':[111],'Genre_Mean_Rating':[5.8],'Director_encoded':[4.5],'Actor1_encoded':[5.3],'Actor2_encoded':[4.5],'Actor3_encoded':[4.5]} # Corrected the keys to match training data features
trail=pd.DataFrame(data)

rating_predicted=model.predict(trail) # Changed 'Model' to 'model' as the variable was named 'model' during model training.
print("Predicted rating:",rating_predicted[0])