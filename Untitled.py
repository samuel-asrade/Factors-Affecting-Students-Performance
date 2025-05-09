#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Import Data
df=pd.read_csv(r'C:\Users\samue\Downloads\Skill Learn\Python Tutorial\factors affecting students performance\student_habits_performance.csv')

# Check for missing values in each column
df.isnull().sum()
# Display rows with missing values
df[df.isnull().any(axis=1)]
# Replace NaN in the 'parental_education_level' column with 'Unknown'
df['parental_education_level'] = df['parental_education_level'].fillna('Unknown')

#Assign dummy variables
df['gender_dummy'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['parjob_dummy'] = df['part_time_job'].map({'No': 0, 'Yes': 1})
df['excurri_dummy'] = df['extracurricular_participation'].map({'No': 0, 'Yes': 1})


#Clustering Data

# Select the features
features = df[['study_hours_per_day', 'sleep_hours', 'social_media_hours', 'attendance_percentage', 'exam_score']]
scaled_features = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)
df.groupby('cluster')[features.columns].mean()
cluster_means = df.groupby('cluster')[features.columns].mean()

#Heatmap representation of the clusters

plt.figure(figsize=(10, 10))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".1f")
plt.title("Average Feature Values per Cluster")
plt.show()


# In[6]:


#Histogram of Exam Score by Gender

bins = np.linspace(min(df['exam_score']), max(df['exam_score']), 5)    # Binning the exam scores into five categories
bin_labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)]  # create ranges of exam scores as the labels
df['exam_score_binned'] = pd.cut(df['exam_score'], bins, labels=bin_labels, include_lowest=True)  # Create the 'exam_score_binned' column
gender_counts = df.groupby(['exam_score_binned', 'gender_dummy'], observed=False).size().unstack(fill_value=0)    # Create a DataFrame
gender_counts['total'] = gender_counts[0] + gender_counts[1] + gender_counts[2]     # Add a 'total' column to sum male and female counts
ax = gender_counts.plot(kind='bar', stacked=False, color=['green', 'yellow', 'red', 'skyblue'], edgecolor='black', figsize=(12, 8))    # Plotting the histogram
# Add the specific number of students on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=10, color='black', 
                xytext=(0, 5), textcoords='offset points')
# Customize the plot
plt.title('Histogram of Exam Score by Gender')
plt.xlabel('Exam Score Range')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(['Male', 'Female', 'Other', 'Total'], loc='upper left')
plt.tight_layout()
plt.show()
print(gender_counts)   # Display the DataFrame with counts for gender and total


# In[14]:


#Histogram of Exam Score by Part-Time Job

bins = np.linspace(min(df['exam_score']), max(df['exam_score']), 5)
bin_labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)]
df['exam_score_binned'] = pd.cut(df['exam_score'], bins, labels=bin_labels, include_lowest=True)
parjob_counts = df.groupby(['exam_score_binned', 'parjob_dummy'], observed=False).size().unstack(fill_value=0)    # Create a DataFrame
parjob_counts['total'] = parjob_counts.sum(axis=1)
ax = parjob_counts.plot(kind='bar', stacked=False, color=['green', 'yellow', 'red'], edgecolor='black', figsize=(14, 10))

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=10, color='black', 
                xytext=(0, 5), textcoords='offset points')
plt.title('Histogram of Exam Score by Part-Time Job')
plt.xlabel('Exam Score Range')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(['Part-time Job: Yes', 'Part-time Job: No', 'Total'], loc='upper left')
plt.tight_layout()
plt.show()
print(parjob_counts)


# In[16]:


# The Effect of Mental Health on Students' Exam Performance

sns.scatterplot(data=df, x='study_hours_per_day', y='exam_score', hue='mental_health_rating')
plt.title('Study Hours vs Exam Score, Colored by Mental Health')
plt.xlabel('Study Hours per Day')
plt.ylabel('Exam Score')
plt.show()


# In[18]:


#Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


lr = LinearRegression()
X = df[['study_hours_per_day', 'social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours', 'mental_health_rating', 'gender_dummy', 'parjob_dummy', 'excurri_dummy']]
Y = df['exam_score']
lr.fit(X, Y)           #fitting the model
Yhat = lr.predict(X)   #prediction
# Output model parameters
print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)


#REGRESSION VISUALIZATION
sns.regplot(x='exam_score', y='study_hours_per_day', data=df)
plt.ylim(0,)
plt.title('Regression Line:Relationship between Sleeping hours and Exam Score')
plt.show()


# In[20]:


# Distribution of actual vs predicted
sns.kdeplot(Y, color='r', label='Actual')
sns.kdeplot(Yhat, color='b', label='Predicted')
plt.legend()
plt.title('Distribution Plot: Actual vs Predicted')
plt.show()


# In[22]:


#Model Evaluation

# Finding the Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Yhat)
print("Mean Squared Error:", mse)

# Calculating the R-squared
r_squared = lr.score(X, Y)
print("R-squared:", r_squared)

#Predicting a single value
prediction = pd.DataFrame({'study_hours_per_day': [5], 'social_media_hours': [2], 'netflix_hours': [1], 'attendance_percentage': [60], 'sleep_hours': [7], 'mental_health_rating': [6], 'gender_dummy': [1], 'parjob_dummy': [1], 'excurri_dummy': [1]})
yhat = lr.predict(prediction)
print("predicted value:", yhat)
print("Coefficient:", lr.coef_)

#Predicting a range of values: We can create a sequence of predicted values by importing numpy
study_hours = np.arange(0, 9, 0.5)
new_input = pd.DataFrame({'study_hours_per_day': study_hours,'social_media_hours': 2, 'netflix_hours': 1,'attendance_percentage': 60,'sleep_hours': 7,'mental_health_rating': 6,'gender_dummy': 1,'parjob_dummy': 1,'excurri_dummy': 1})
yhat = lr.predict(new_input)
yhat = np.clip(yhat, 0, 100)  # Ensures predictions of exam scores are between 0 and 100
print(yhat)


# Plotting the result in a line graph
plt.figure(figsize=(12, 8))
plt.plot(new_input['study_hours_per_day'], yhat, color='red', label='Predicted Exam Score')
plt.scatter(df['study_hours_per_day'], df['exam_score'], color='blue', alpha=0.5, label='Actual Data')  # Actual data points

# Customize plot
plt.title('Predicted Exam Score vs. Study Hours')
plt.xlabel('Study Hours per Day')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


# Model Validation
    #It helps to visually and statistically assess how well the model fits the training data and generalizes to unseen testing data.

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)     # Split data into train(70%) and test (30%) data

# Train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Distribution plot
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)
sns.kdeplot(y_train, label='Actual Train', fill=True)
sns.kdeplot(y_train_pred, label='Predicted Train', fill=True)
sns.kdeplot(y_test, label='Actual Test', fill=True)
sns.kdeplot(y_test_pred, label='Predicted Test', fill=True)
plt.title("Distribution Plot: Actual vs Predicted (Train & Test)")
plt.legend()
plt.show()


# In[26]:


# Cross-validation
 #to check if polynomial regression could fit into the data, the following is done

from sklearn.metrics import r2_score

scores = cross_val_score(lr, X, Y, cv=3)
print("Cross-validation score:", np.mean(scores))
yhat_cv = cross_val_predict(lr, X, Y, cv=3)

# Polynomial regression - checking for underfitting/overfitting
Rsqu_test = []
order = [1, 2, 3, 4]

for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train)
    x_test_pr = pr.transform(x_test)
    lr.fit(x_train_pr, y_train)
    Rsqu_test.append(lr.score(x_test_pr, y_test))

print("Polynomial R-squared test scores:", Rsqu_test)


# In[ ]:




