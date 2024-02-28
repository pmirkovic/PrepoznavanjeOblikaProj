import pandas as pd
import matplotlib.pyplot as plt

# Insert data from csv file
df = pd.read_csv("CO2 Emissions_Canada.csv")
print("\nShow first 5 elements:\n", df.head())

# Basic information about dataset
print("\nNumber of samples: ", df.shape[0])
print("\nNumber of features: ", df.shape[1])
print("\nInformation about features: \n", df.dtypes)

#Categorical features
print('Make', df['Make'].unique())
print('Model', df['Model'].unique())
print('Vehicle Class', df['Vehicle Class'].unique())
print('Transmission', df['Transmission'].unique())
print('Fuel Type', df['Fuel Type'].unique())

#Check for missing values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/len(df)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(12))