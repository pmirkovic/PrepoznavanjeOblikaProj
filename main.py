import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import skew, kurtosis


#3
# učitavanje u dataframe format
df = pd.read_csv('CO2 Emissions_Canada.csv')
# Prikaz prvih 5 uzoraka
print(df.head())



#4
#Koliko ima uzoraka
print(df.shape)
#Koliko ima obelezja
print(df.dtypes)
#Nedostajuci podaci
print(df.isnull().sum() / df.shape[0] * 100)

print(df["Fuel Type"].unique())

# Računanje opsega
co2_range = df['CO2 Emissions(g/km)'].max() - df['CO2 Emissions(g/km)'].min()

# Računanje srednje vrednosti
co2_mean = df['CO2 Emissions(g/km)'].mean()

# Računanje mediane
co2_median = df['CO2 Emissions(g/km)'].median()

# Ispis rezultata
print(f"Opseg CO2: {co2_range}")
print(f"Srednja vrednost CO2: {co2_mean}")
print(f"Medijana CO2: {co2_median}")
print(df['CO2 Emissions(g/km)'].max())
print(df['CO2 Emissions(g/km)'].min())

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Provera nevalidnih vrednosti po numeričkim kolonama
for column in numeric_columns:
    invalid_values = df[df[column].isnull() | (df[column] < 0)]  # Provera nedostajućih ili negativnih vrednosti
    if not invalid_values.empty:
        print(f"Nevalidne vrednosti u koloni '{column}':")
        print(invalid_values)

categorical_columns = df.select_dtypes(include=['object']).columns

# Provera nevalidnih vrednosti po kategoričkim kolonama
for column in categorical_columns:
    invalid_values = df[df[column].isnull()]  # Provera nedostajućih vrednosti
    if not invalid_values.empty:
        print(f"Nevalidne vrednosti u koloni '{column}':")
        print(invalid_values)

# Provera autlajera po numeričkim kolonama
for column in numeric_columns:
    q1 = df[column].quantile(0.25)  # Donji kvartil
    q3 = df[column].quantile(0.75)  # Gornji kvartil
    iqr = q3 - q1  # Interkvartilni raspon
    lower_bound = q1 - 1.5 * iqr  # Donja granica
    upper_bound = q3 + 1.5 * iqr  # Gornja granica

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    if not outliers.empty:
        print(f"Autlajeri u koloni '{column}':")
        print(outliers)

# Vizualizacija autlajera
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    plt.title(f'Boxplot za kolonu "{column}"')
    plt.boxplot(df[column])
    plt.ylabel('Vrednosti')
    plt.xlabel('Kolona')
    plt.show()

# Pretvaranje kategoričkih obeležja u numeričke pomoću One-Hot Encodinga
data_encoded = pd.get_dummies(df, columns=['Make', 'Model', 'Vehicle Class','Transmission','Fuel Type'])


# Izračunavanje korelacija
correlation_matrix = data_encoded.corr().abs()

# Identifikacija parova obeležja sa korelacijom većom od 0.7
high_correlation = (correlation_matrix > 0.7) & (correlation_matrix < 1.0)
correlated_features = (high_correlation).any()

if correlated_features.any():
    print("Parovi obelezja sa korelacijom većom od 0.7:")
    print(correlation_matrix.loc[correlated_features, correlated_features])
else:
    print("Nema parova obelezja sa korelacijom većom od 0.7.")


# Prikaz heatmap-a
plt.figure(figsize=(10, 8))
sb.heatmap(correlation_matrix.loc[correlated_features, correlated_features], annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa toplotnih mapa korelacije')
plt.show()

# Izdvajanje varijable CO2
co2_variable = df['CO2 Emissions(g/km)']

# Izračunavanje asimetrije i spljoštenosti
skewness = skew(co2_variable)
kurt = kurtosis(co2_variable)
# Assuming 'data' is your DataFrame and 'FeatureName' is the column you want to visualize
sb.set(style="whitegrid")  # Set the style
plt.figure(figsize=(8, 6))  # Set the size of the plot
# Create the distribution plot for the 'FeatureName'
sb.histplot(df['CO2 Emissions(g/km)'], kde=True, color='skyblue')

# Set labels and title
plt.xlabel('CO2 Emissions(g/km)')
plt.ylabel('Frequency')
plt.title('Distribution Plot of FeatureName')

plt.show()  # Show the plot
# Prikaz histograma varijable CO2
plt.figure(figsize=(8, 6))
plt.hist(co2_variable, bins=30, alpha=0.7, color='skyblue')
plt.xlabel('CO2 Emissions(g/km)')
plt.ylabel('Frequency')
plt.title('Distribution of CO2 Emissions')
plt.axvline(co2_variable.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {co2_variable.mean():.2f}')
plt.legend()

# Dodavanje informacija o asimetriji i spljoštenosti na grafikon
plt.text(0.05, 0.85, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'Kurtosis: {kurt:.2f}', transform=plt.gca().transAxes)

plt.show()