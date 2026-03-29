import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


#Loading
df = pd.read_csv("Dataset/DataSet.csv") #Loads the csv(excel) file in python

#Understanding
print("Shape: ",df.shape) #Shows the no. of columns and rows in the dataset
print("\nFirst 5 rows:\n") #Extra print statement to prevent big data from getting truncated
print(df.head())

print("\nColumns:")  #Lists the columns
for i in df.columns:
	print(i) 

print("\nMissing values:") #Lists the  missing values 
print(df.isnull().sum())

print("\nTarget distribution:")
print(df['fraudulent'].value_counts(),"\n\n",df['fraudulent'].value_counts(normalize=True)) #Checks the class distributions and shows the count and %age

#Cleaning
df = df.drop(['in_balanced_dataset'], axis=1) #Removed extra and useless identifier

df['fraudulent'] = df['fraudulent'].map({'f': 0, 't': 1}) #Mapping False(f) to 0 and True(t) to 1
df['telecommuting'] = df['telecommuting'].map({'f': 0, 't': 1})
df['has_company_logo'] = df['has_company_logo'].map({'f': 0, 't': 1})
df['has_questions'] = df['has_questions'].map({'f': 0, 't': 1})

df=df.fillna('') #Handling missing values 

df['description']=df['description'].str.lower() #Convert text to lowercase
df['requirements'] = df['requirements'].str.lower()
df['company_profile'] = df['company_profile'].str.lower()

df['desc_length'] = df['description'].apply(len) #Feature
df['company_profile_length'] = df['company_profile'].apply(len)
df['requirements_length'] = df['requirements'].apply(len)
df['title_length'] = df['title'].apply(len)

df['has_company_profile'] = (df['company_profile'] != '').astype(int) #Feature
df['has_salary'] = (df['salary_range'] != '').astype(int)
df['has_requirements'] = (df['requirements'] != '').astype(int)

keywords = ['easy money', 'quick cash', 'no experience', 'work from home'] #Keyword feature
for word in keywords:
    df[word] = df['description'].str.contains(word).astype(int)

df = df.drop(['title', 'location', 'department','salary_range', 'company_profile','description', 'requirements', 'benefits','employment_type','required_experience','required_education','industry','function'],axis=1) #Removing text columns

print(df.dtypes) #To check everything is in int or float 

df_majority = df[df.fraudulent == 0] #Upsampling
df_minority = df[df.fraudulent == 1]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
df = pd.concat([df_majority, df_minority_upsampled])
df = df.sample(frac=1, random_state=42) 

X = df.drop('fraudulent', axis=1) #Spliting data 0.8,0.2
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=10, random_state=42) #Traning
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)) #Evaluation
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20,10)) #Decision tree plotted
plot_tree(model, feature_names=X.columns, filled=True)
plt.savefig("decision_tree.png", dpi=300, bbox_inches='tight')
plt.show()

importance = pd.DataFrame({'Feature': X.columns,'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

print(importance)

# cd python2\DMDW project
# python Fake_job_project.py



