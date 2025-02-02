import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


#Učitavanje u dataframe format
df = pd.read_csv('CO2 Emissions_Canada.csv')

# #Prikaz prvih 5 uzoraka
# print(df.head())
#
# #Koliko ima uzoraka
# print(df.shape)
#
# #Koliko ima obelezja
# print(df.dtypes)
#
# #Nedostajuci podaci
# print(df.isnull().sum() / df.shape[0] * 100)
# print(df["Fuel Type"].unique())
#
# #Računanje opsega
# co2_range = df['CO2 Emissions(g/km)'].max() - df['CO2 Emissions(g/km)'].min()
#
# #Računanje srednje vrednosti
# co2_mean = df['CO2 Emissions(g/km)'].mean()
#
# #Računanje mediane
# co2_median = df['CO2 Emissions(g/km)'].median()
#
# #Ispis rezultata
# print(f"Opseg CO2: {co2_range}")
# print(f"Srednja vrednost CO2: {co2_mean}")
# print(f"Medijana CO2: {co2_median}")
# print(df['CO2 Emissions(g/km)'].max())
# print(df['CO2 Emissions(g/km)'].min())

# plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
#
# # Get the top 5 models with highest average CO2 emissions
# top_5_co2 = (
#     df.groupby("Make")["CO2 Emissions(g/km)"].mean().sort_values(ascending=False).head(5)
# )
#
# # Plot the bar chart
# top_5_co2.plot(kind='bar')
#
# plt.title("Prosečna emisija CO2 po 5 modela automobila s najvećim emisijama")
# plt.xlabel("Modeli automobila")
# plt.ylabel("Prosečna emisija CO2")
# plt.xticks(rotation=90)  # Rotate x-axis labels if needed
# plt.show()


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
# #Provera nevalidnih vrednosti po kategoričkim kolonama
# for column in categorical_columns:
#     invalid_values = df[df[column].isnull()]  # Provera nedostajućih vrednosti
#     if not invalid_values.empty:
#         print(f"Nevalidne vrednosti u koloni '{column}':")
#         print(invalid_values)

# #Provera autlajera po numeričkim kolonama
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

# #Vizualizacija autlajera
# for column in numeric_columns:
#     plt.figure(figsize=(8, 6))
#     plt.title(f'Boxplot za kolonu "{column}"')
#     plt.boxplot(df[column])
#     plt.ylabel('Vrednosti')
#     plt.xlabel('Kolona')
#     plt.show()

#Pretvaranje kategoričkih obeležja u numeričke pomoću One-Hot Encodinga
data_encoded = pd.get_dummies(df, columns=['Make', 'Model', 'Vehicle Class','Transmission','Fuel Type'])


# #Izračunavanje korelacija
# correlation_matrix = data_encoded.corr().abs()
#
# #Identifikacija parova obeležja sa korelacijom većom od 0.7
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
# #Prikaz heatmap-a
# plt.figure(figsize=(10, 8))
# sb.heatmap(correlation_matrix.loc[correlated_features, correlated_features], annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Mapa toplotnih mapa korelacije')
# plt.show()
#
# #Izdvajanje varijable CO2
# co2_variable = df['CO2 Emissions(g/km)']
#
# #Izračunavanje asimetrije i spljoštenosti
# skewness = skew(co2_variable)
# kurt = kurtosis(co2_variable)
# sb.set(style="whitegrid")
# plt.figure(figsize=(8, 6))
# sb.histplot(df['CO2 Emissions(g/km)'], kde=True, color='skyblue')
# plt.xlabel('CO2 Emissions(g/km)')
# plt.ylabel('Frequency')
# plt.title('Distribution Plot of FeatureName')
# plt.show()
#
# #Prikaz histograma varijable CO2
# plt.figure(figsize=(8, 6))
# plt.hist(co2_variable, bins=30, alpha=0.7, color='skyblue')
# plt.xlabel('CO2 Emissions(g/km)')
# plt.ylabel('Frequency')
# plt.title('Distribution of CO2 Emissions')
# plt.axvline(co2_variable.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {co2_variable.mean():.2f}')
# plt.legend()
#
# #Dodavanje informacija o asimetriji i spljoštenosti na grafikon
# plt.text(0.05, 0.85, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes)
# plt.text(0.05, 0.8, f'Kurtosis: {kurt:.2f}', transform=plt.gca().transAxes)
# plt.show()

df.loc[df['Fuel Type'] == 'Z','Fuel Type'] = 0
df.loc[df['Fuel Type'] == 'D','Fuel Type'] = 1
df.loc[df['Fuel Type'] == 'X','Fuel Type'] = 2
df.loc[df['Fuel Type'] == 'E','Fuel Type'] = 3
df.loc[df['Fuel Type'] == 'N','Fuel Type'] = 4

#################################################################################################################

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

#Trening i test skup
x_train1, x_test, y_train1, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size=0.1, random_state=42)


# #Provera najboljeg k
# for k in range(1, 10):
#     knn_regressor = KNeighborsRegressor(n_neighbors=k)
#     knn_regressor.fit(x_train, y_train)
#     print(f'Score for k={k}: {knn_regressor.score(x_test, y_test)}')
# print("\n")

# Standardizacija podataka
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

#Linearna regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_Linear = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    linear_model = LinearRegression()
    linear_model.fit(x_train_fold, y_train_fold)

    y_pred_val = linear_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Linear.append(mae_val)

# Ispis rezultata
print("MAE linearna regresija:", np.mean(cross_val_results_Linear))

#Lasso regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_Lasso = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    lasso_model = Lasso(alpha=5)
    lasso_model.fit(x_train_fold, y_train_fold)

    y_pred_val = lasso_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Lasso.append(mae_val)

# Ispis rezultata
print("MAE lasso regresija:", np.mean(cross_val_results_Lasso))

#Ridge regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_Ridge = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    ridge_model = Ridge(alpha=5)
    ridge_model.fit(x_train_fold, y_train_fold)

    y_pred_val = ridge_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Ridge.append(mae_val)

# Ispis rezultata
print("MAE ridge regresija:", np.mean(cross_val_results_Ridge))

#KNN regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_KNN = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    knn_model = KNeighborsRegressor(n_neighbors=2)
    knn_model.fit(x_train_fold, y_train_fold)

    y_pred_val = knn_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_KNN.append(mae_val)

# Ispis rezultata
print("MAE knn regresija:", np.mean(cross_val_results_KNN))

#Random forest regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_RadnomForest = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    RandomForest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    RandomForest_model.fit(x_train_fold, y_train_fold)

    y_pred_val = RandomForest_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_RadnomForest.append(mae_val)

# Ispis rezultata
print("MAE Radnom forest regresija:", np.mean(cross_val_results_RadnomForest))

print("\nRedukcija putem pca\n")
# Inicijalizacija PCA modela sa željenim brojem komponenti
pca = PCA(n_components=2)

# Fitovanje PCA modela na standardizovanim podacima i transformacija podataka
x_train = pca.fit_transform(x_train)

# PCA transformacija testnih podataka
x_val = pca.transform(x_val)


#Linearna regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_Linear = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    linear_model = LinearRegression()
    linear_model.fit(x_train_fold, y_train_fold)

    y_pred_val = linear_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Linear.append(mae_val)

# Ispis rezultata
print("MAE linearna regresija:", np.mean(cross_val_results_Linear))

#Lasso regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_Lasso = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    lasso_model = Lasso(alpha=5)
    lasso_model.fit(x_train_fold, y_train_fold)

    y_pred_val = lasso_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Lasso.append(mae_val)

# Ispis rezultata
print("MAE lasso regresija:", np.mean(cross_val_results_Lasso))

#Ridge regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_Ridge = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    ridge_model = Ridge(alpha=5)
    ridge_model.fit(x_train_fold, y_train_fold)

    y_pred_val = ridge_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Ridge.append(mae_val)

# Ispis rezultata
print("MAE ridge regresija:", np.mean(cross_val_results_Ridge))

#KNN regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_KNN = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    knn_model = KNeighborsRegressor(n_neighbors=2)
    knn_model.fit(x_train_fold, y_train_fold)

    y_pred_val = knn_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_KNN.append(mae_val)

# Ispis rezultata
print("MAE knn regresija:", np.mean(cross_val_results_KNN))

#Random forest regresija

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Prikupljanje rezultata unakrsne validacije
cross_val_results_RadnomForest = []

# Petlja kroz foldove
for train_index, test_index in kf.split(x_train, y_train):
    # Podela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train[train_index, :], x_train[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog modela
    RandomForest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    RandomForest_model.fit(x_train_fold, y_train_fold)

    y_pred_val = RandomForest_model.predict(x_val_fold)

    # Izračunavanje MAE
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_RadnomForest.append(mae_val)

# Ispis rezultata
print("MAE Radnom forest regresija:", np.mean(cross_val_results_RadnomForest))

#Ridge regreisja
Ridge_model = Ridge(alpha=5)
Ridge_model.fit(x_train1, y_train1)
y_pred_val_ridge = Ridge_model.predict(x_test)
print("\nRidge regresija evaluacija na test skup\n")
model_evaluation(y_test, y_pred_val_ridge, x_train1.shape[0], x_train1.shape[1])

#KNN regresija
knn_regressor = KNeighborsRegressor(n_neighbors=2)
knn_regressor.fit(x_train1, y_train1)
y_pred_test_knn = knn_regressor.predict(x_test)
print("\nKNN regresija evaluacija na test skup\n")
model_evaluation(y_test, y_pred_test_knn, x_train1.shape[0], x_train1.shape[1])

#Random forest regresija
RandomForest_tree = RandomForestRegressor(n_estimators=100, random_state=42)
RandomForest_tree.fit(x_train1, y_train1)
#Predikcija na testnom skupu
y_pred_test_RF = RandomForest_tree.predict(x_test)
print("\nRadnom forest regresija evaluacija na test skup\n")
model_evaluation(y_test, y_pred_test_RF, x_train1.shape[0], x_train1.shape[1])

