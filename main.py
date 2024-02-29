import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#3
# učitavanje u dataframe format
df = pd.read_csv('CO2 Emissions_Canada.csv')
# Prikaz prvih 5 uzoraka
print(df.head())


#
# #4
# #Koliko ima uzoraka
# print(df.shape)
# #Koliko ima obelezja
# print(df.dtypes)
# #Nedostajuci podaci
# print(df.isnull().sum() / df.shape[0] * 100)
#
# print(df["Fuel Type"].unique())
#
# # Računanje opsega
# co2_range = df['CO2 Emissions(g/km)'].max() - df['CO2 Emissions(g/km)'].min()
#
# # Računanje srednje vrednosti
# co2_mean = df['CO2 Emissions(g/km)'].mean()
#
# # Računanje mediane
# co2_median = df['CO2 Emissions(g/km)'].median()
#
# # Ispis rezultata
# print(f"Opseg CO2: {co2_range}")
# print(f"Srednja vrednost CO2: {co2_mean}")
# print(f"Medijana CO2: {co2_median}")
# print(df['CO2 Emissions(g/km)'].max())
# print(df['CO2 Emissions(g/km)'].min())
#
# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
#
# # Provera nevalidnih vrednosti po numeričkim kolonama
# for column in numeric_columns:
#     invalid_values = df[df[column].isnull() | (df[column] < 0)]  # Provera nedostajućih ili negativnih vrednosti
#     if not invalid_values.empty:
#         print(f"Nevalidne vrednosti u koloni '{column}':")
#         print(invalid_values)
#
# categorical_columns = df.select_dtypes(include=['object']).columns
#
# # Provera nevalidnih vrednosti po kategoričkim kolonama
# for column in categorical_columns:
#     invalid_values = df[df[column].isnull()]  # Provera nedostajućih vrednosti
#     if not invalid_values.empty:
#         print(f"Nevalidne vrednosti u koloni '{column}':")
#         print(invalid_values)
#
# # Provera autlajera po numeričkim kolonama
# for column in numeric_columns:
#     q1 = df[column].quantile(0.25)  # Donji kvartil
#     q3 = df[column].quantile(0.75)  # Gornji kvartil
#     iqr = q3 - q1  # Interkvartilni raspon
#     lower_bound = q1 - 1.5 * iqr  # Donja granica
#     upper_bound = q3 + 1.5 * iqr  # Gornja granica
#
#     outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
#     if not outliers.empty:
#         print(f"Autlajeri u koloni '{column}':")
#         print(outliers)
#
# # Vizualizacija autlajera
# for column in numeric_columns:
#     plt.figure(figsize=(8, 6))
#     plt.title(f'Boxplot za kolonu "{column}"')
#     plt.boxplot(df[column])
#     plt.ylabel('Vrednosti')
#     plt.xlabel('Kolona')
#     plt.show()
#
# # Pretvaranje kategoričkih obeležja u numeričke pomoću One-Hot Encodinga
# data_encoded = pd.get_dummies(df, columns=['Make', 'Model', 'Vehicle Class','Transmission','Fuel Type'])
#
#
# # Izračunavanje korelacija
# correlation_matrix = data_encoded.corr().abs()
#
# # Identifikacija parova obeležja sa korelacijom većom od 0.7
# high_correlation = (correlation_matrix > 0.7) & (correlation_matrix < 1.0)
# correlated_features = (high_correlation).any()
#
# if correlated_features.any():
#     print("Parovi obelezja sa korelacijom većom od 0.7:")
#     print(correlation_matrix.loc[correlated_features, correlated_features])
# else:
#     print("Nema parova obelezja sa korelacijom većom od 0.7.")
#
#
# # # Prikaz heatmap-a
# # plt.figure(figsize=(10, 8))
# # sb.heatmap(correlation_matrix.loc[correlated_features, correlated_features], annot=True, cmap='coolwarm', fmt='.2f')
# # plt.title('Mapa toplotnih mapa korelacije')
# # plt.show()
#
# # Izdvajanje varijable CO2
# co2_variable = df['CO2 Emissions(g/km)']
#
# # Izračunavanje asimetrije i spljoštenosti
# skewness = skew(co2_variable)
# kurt = kurtosis(co2_variable)
# # Assuming 'data' is your DataFrame and 'FeatureName' is the column you want to visualize
# sb.set(style="whitegrid")  # Set the style
# plt.figure(figsize=(8, 6))  # Set the size of the plot
# # Create the distribution plot for the 'FeatureName'
# sb.histplot(df['CO2 Emissions(g/km)'], kde=True, color='skyblue')
#
# # Set labels and title
# plt.xlabel('CO2 Emissions(g/km)')
# plt.ylabel('Frequency')
# plt.title('Distribution Plot of FeatureName')
#
# plt.show()  # Show the plot
# # Prikaz histograma varijable CO2
# plt.figure(figsize=(8, 6))
# plt.hist(co2_variable, bins=30, alpha=0.7, color='skyblue')
# plt.xlabel('CO2 Emissions(g/km)')
# plt.ylabel('Frequency')
# plt.title('Distribution of CO2 Emissions')
# plt.axvline(co2_variable.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {co2_variable.mean():.2f}')
# plt.legend()
#
# # Dodavanje informacija o asimetriji i spljoštenosti na grafikon
# plt.text(0.05, 0.85, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes)
# plt.text(0.05, 0.8, f'Kurtosis: {kurt:.2f}', transform=plt.gca().transAxes)
#
# plt.show()



df.loc[df['Fuel Type'] == 'Z','Fuel Type'] = 0
df.loc[df['Fuel Type'] == 'D','Fuel Type'] = 1
df.loc[df['Fuel Type'] == 'X','Fuel Type'] = 2
df.loc[df['Fuel Type'] == 'E','Fuel Type'] = 3
df.loc[df['Fuel Type'] == 'N','Fuel Type'] = 4

X = df.drop(['CO2 Emissions(g/km)','Make','Model','Vehicle Class','Transmission'], axis=1).copy()
y = df['CO2 Emissions(g/km)'].copy()


def model_evaluation(y_test, y_predicted, N, d):
    mse = np.mean((y_test-y_predicted)**2)
    # mse = mean_squared_error(y_test,y_predicted)
    mae = np.mean(np.abs(y_test-y_predicted))
    # mae = mean_absolute_error(y_test,y_predicted)
    rmse = np.sqrt(mse)
    r2 = 1-np.sum((y_test-y_predicted)**2)/np.sum((y_test-np.mean(y_test))**2)
    # r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-((1-r2)*(N-1))/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)

    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y_test.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))
    return mse,mae,rmse,r2,r2_adj

#Linearna regresija
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.8, random_state=42)
x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

kf = KFold(n_splits=10,shuffle=True,random_state=42)
indexes = kf.split(X,y)
for train_index, test_index in indexes:
    x_train = X.iloc[train_index,:]
    x_test = X.iloc[test_index,:]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("\nLinearna regresiaj")
    linear_model = LinearRegression()
    #obuka
    linear_model.fit(x_train,y_train)
    #testiranje
    y_pred = linear_model.predict(x_test)

    model_evaluation(y_test,y_pred,x_train.shape[0],x_train.shape[1])
    print("koeficijenti: ", linear_model.coef_)

    print("\nRidge regresiaj")
    ridge_model = Ridge(alpha=5)
    # obuka
    ridge_model.fit(x_train, y_train)
    # testiranje
    y_ridge_pred = ridge_model.predict(x_test)

    model_evaluation(y_test, y_ridge_pred, x_train.shape[0], x_train.shape[1])
    print("koeficijenti: ", ridge_model.coef_)

    print("\nLasso regresiaj")
    lasso_model = Lasso(alpha=0.01)
    # obuka
    lasso_model.fit(x_train, y_train)
    # testiranje
    y_lasso_pred = lasso_model.predict(x_test)

    model_evaluation(y_test, y_lasso_pred, x_train.shape[0], x_train.shape[1])
    print("koeficijenti: ", lasso_model.coef_)




