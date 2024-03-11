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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, random_state=42)

#Provera najboljeg k
# for k in range(1, 10):
#     knn_regressor = KNeighborsRegressor(n_neighbors=k)
#     knn_regressor.fit(x_train, y_train)
#     print(f'Score for k={k}: {knn_regressor.score(x_test, y_test)}')

#################################linearna regresija#####################################
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_results_Linear = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Skaliranje značajki
    scaler = StandardScaler()
    x_train_fold_scaled = scaler.fit_transform(x_train_fold)
    x_val_fold_scaled = scaler.transform(x_val_fold)

    # Inicijalizacija i treniranje linearnog regresijskog modela
    linear_model = LinearRegression()
    linear_model.fit(x_train_fold_scaled, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = linear_model.predict(x_val_fold_scaled)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_results_Linear.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_results_Linear.index(min(cross_val_results_Linear))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train.iloc[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Skaliranje trening podataka
scaler = StandardScaler()
best_x_train_fold_scaled = scaler.fit_transform(best_x_train_fold)

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
linear_model = LinearRegression()
linear_model.fit(best_x_train_fold_scaled, best_y_train_fold)

# Skaliranje testnih podataka
x_test_scaled = scaler.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = linear_model.predict(x_test_scaled)
model_evaluation(y_test, y_pred_test, x_train.shape[0], x_train.shape[1])
# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu Linearna regresija:", mae_test)


####################################Ridge###################################################
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsRidge = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Skaliranje značajki
    scaler = StandardScaler()
    x_train_fold_scaled = scaler.fit_transform(x_train_fold)
    x_val_fold_scaled = scaler.transform(x_val_fold)

    # Inicijalizacija i treniranje linearnog regresijskog modela
    Ridge_model = Ridge(alpha=5)
    Ridge_model.fit(x_train_fold_scaled, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = Ridge_model.predict(x_val_fold_scaled)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsRidge.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsRidge.index(min(cross_val_resultsRidge))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train.iloc[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Skaliranje trening podataka
scaler = StandardScaler()
best_x_train_fold_scaled = scaler.fit_transform(best_x_train_fold)

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
Ridge_model = Ridge(alpha=5)
Ridge_model.fit(best_x_train_fold_scaled, best_y_train_fold)

# Skaliranje testnih podataka
x_test_scaled = scaler.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = Ridge_model.predict(x_test_scaled)

# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu Ridge regresija:", mae_test)
#######################################################################################

#####################################Lasso##################################################
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsLasso = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Skaliranje značajki
    scaler = StandardScaler()
    x_train_fold_scaled = scaler.fit_transform(x_train_fold)
    x_val_fold_scaled = scaler.transform(x_val_fold)

    # Inicijalizacija i treniranje linearnog regresijskog modela
    Lasso_model  = Lasso(alpha=0.01)
    Lasso_model.fit(x_train_fold_scaled, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = Lasso_model.predict(x_val_fold_scaled)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsLasso.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsLasso.index(min(cross_val_resultsLasso))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train.iloc[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Skaliranje trening podataka
scaler = StandardScaler()
best_x_train_fold_scaled = scaler.fit_transform(best_x_train_fold)

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
Lasso_model  = Lasso(alpha=0.01)
Lasso_model.fit(best_x_train_fold_scaled, best_y_train_fold)

# Skaliranje testnih podataka
x_test_scaled = scaler.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = Lasso_model.predict(x_test_scaled)

# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu Lasso regresija:", mae_test)
############################################################################################

######################################Knn_regressor#######################################################
print("\nPrikaz modela evaluacije knn regresije\n")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsKNN = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Skaliranje značajki
    scaler = StandardScaler()
    x_train_fold_scaled = scaler.fit_transform(x_train_fold)
    x_val_fold_scaled = scaler.transform(x_val_fold)

    # Inicijalizacija i treniranje linearnog regresijskog modela
    knn_regressor = KNeighborsRegressor(n_neighbors=2)
    knn_regressor.fit(x_train_fold_scaled, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = knn_regressor.predict(x_val_fold_scaled)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsKNN.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsKNN.index(min(cross_val_resultsKNN))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train.iloc[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Skaliranje trening podataka
scaler = StandardScaler()
best_x_train_fold_scaled = scaler.fit_transform(best_x_train_fold)

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
knn_regressor = KNeighborsRegressor(n_neighbors=2)
knn_regressor.fit(best_x_train_fold_scaled, best_y_train_fold)

# Skaliranje testnih podataka
x_test_scaled = scaler.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = knn_regressor.predict(x_test_scaled)
model_evaluation(y_test, y_pred_test, x_train.shape[0], x_train.shape[1])
# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu KNN regresija:", mae_test)
#############################################################################################

##################################RandomForest_tree###########################################################
print("\nPrikaz modela evaluacije radnom forest regresije\n")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsRadnomForest = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Skaliranje značajki
    scaler = StandardScaler()
    x_train_fold_scaled = scaler.fit_transform(x_train_fold)
    x_val_fold_scaled = scaler.transform(x_val_fold)

    # Inicijalizacija i treniranje linearnog regresijskog modela
    RandomForest_tree = RandomForestRegressor(n_estimators=100, random_state=42)
    RandomForest_tree.fit(x_train_fold_scaled, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = RandomForest_tree.predict(x_val_fold_scaled)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsRadnomForest.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsRadnomForest.index(min(cross_val_resultsRadnomForest))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train.iloc[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Skaliranje trening podataka
scaler = StandardScaler()
best_x_train_fold_scaled = scaler.fit_transform(best_x_train_fold)

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
RandomForest_tree = RandomForestRegressor(n_estimators=100, random_state=42)
RandomForest_tree.fit(best_x_train_fold_scaled, best_y_train_fold)

# Skaliranje testnih podataka
x_test_scaled = scaler.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = RandomForest_tree.predict(x_test_scaled)
model_evaluation(y_test, y_pred_test, x_train.shape[0], x_train.shape[1])
# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu RandomForest regresija:", mae_test)
#############################################################################################

#print("\nRedukcija putem PCA\n")

############################Redukcija pomocu PCA#######################################
# Inicijalizacija PCA
pca = PCA(n_components=7)

# Primjena PCA na trening skupu
x_train_pca = pca.fit_transform(x_train)


#################################linearna regresija#####################################

# Inicijalizacija KFold unakrsne validacije
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsLinearPCA = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train_pca, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train_pca[train_index, :], x_train_pca[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog regresijskog modela
    linear_model = LinearRegression()
    linear_model.fit(x_train_fold, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = linear_model.predict(x_val_fold)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsLinearPCA.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsLinearPCA.index(min(cross_val_resultsLinearPCA))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train_pca, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train_pca[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
linear_model = LinearRegression()
linear_model.fit(best_x_train_fold, best_y_train_fold)

# Primjena PCA na testni skup
x_test_pca = pca.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = linear_model.predict(x_test_pca)

# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu Linearna regresija s PCA:", mae_test)
######################################################################################

#################################Ridge#####################################

# Inicijalizacija KFold unakrsne validacije
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsRidgePCA = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train_pca, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train_pca[train_index, :], x_train_pca[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog regresijskog modela
    ridge_model = Ridge(alpha=5)
    ridge_model.fit(x_train_fold, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = ridge_model.predict(x_val_fold)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsRidgePCA.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsRidgePCA.index(min(cross_val_resultsRidgePCA))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train_pca, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train_pca[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
ridge_model = Ridge(alpha=5)
ridge_model.fit(best_x_train_fold, best_y_train_fold)

# Primjena PCA na testni skup
x_test_pca = pca.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = ridge_model.predict(x_test_pca)

# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu Ridge regresija s PCA:", mae_test)
######################################################################################

#################################Lasso#####################################

# Inicijalizacija KFold unakrsne validacije
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsLassoPCA = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train_pca, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train_pca[train_index, :], x_train_pca[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog regresijskog modela
    lasso_model = Lasso(alpha=0.01)
    lasso_model.fit(x_train_fold, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = lasso_model.predict(x_val_fold)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsLassoPCA.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsLassoPCA.index(min(cross_val_resultsLassoPCA))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train_pca, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train_pca[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(best_x_train_fold, best_y_train_fold)

# Primjena PCA na testni skup
x_test_pca = pca.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = lasso_model.predict(x_test_pca)

# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu Lasso regresija s PCA:", mae_test)
######################################################################################

#################################KNN#####################################

# Inicijalizacija KFold unakrsne validacije
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsKNNPCA = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train_pca, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train_pca[train_index, :], x_train_pca[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog regresijskog modela
    knn_regressor = KNeighborsRegressor(n_neighbors=2)
    knn_regressor.fit(x_train_fold, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = knn_regressor.predict(x_val_fold)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsKNNPCA.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsKNNPCA.index(min(cross_val_resultsKNNPCA))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train_pca, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train_pca[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
knn_regressor = KNeighborsRegressor(n_neighbors=2)
knn_regressor.fit(best_x_train_fold, best_y_train_fold)

# Primjena PCA na testni skup
x_test_pca = pca.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = knn_regressor.predict(x_test_pca)

# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu KNN regresija s PCA:", mae_test)
######################################################################################

#################################RandomForest#####################################
print("\nPrikaz modela evaluacije radnom forest regresije sa redukcijom pomocu pca\n")
# Inicijalizacija KFold unakrsne validacije
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prikupljanje rezultata unakrsne validacije
cross_val_resultsRadnomForestPCA = []

# Petlja kroz preklope
for train_index, test_index in kf.split(x_train_pca, y_train):
    # Podjela trening skupa na podskup za treniranje i validaciju
    x_train_fold, x_val_fold = x_train_pca[train_index, :], x_train_pca[test_index, :]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Inicijalizacija i treniranje linearnog regresijskog modela
    RandomForest_tree = RandomForestRegressor(n_estimators=100, random_state=42)
    RandomForest_tree.fit(x_train_fold, y_train_fold)

    # Predikcija na validacionom skupu
    y_pred_val = RandomForest_tree.predict(x_val_fold)

    # Evaluacija modela na validacionom skupu
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    cross_val_resultsRadnomForestPCA.append(mae_val)

# Indeks najboljeg preklopa (najmanji MAE)
best_fold_index = cross_val_resultsRadnomForestPCA.index(min(cross_val_resultsRadnomForestPCA))

# Dobivanje indeksa za najbolji preklop
best_fold_indexes = next(kf.split(x_train_pca, y_train), None)

if best_fold_indexes:
    train_index, _ = best_fold_indexes
    best_x_train_fold, best_y_train_fold = x_train_pca[train_index, :], y_train.iloc[train_index]
else:
    print("Nije pronađen indeks najboljeg preklopa.")

# Treniranje modela na cijelom trening skupu koristeći najbolji preklop
RandomForest_tree = RandomForestRegressor(n_estimators=100, random_state=42)
RandomForest_tree.fit(best_x_train_fold, best_y_train_fold)

# Primjena PCA na testni skup
x_test_pca = pca.transform(x_test)

# Predikcija na testnom skupu
y_pred_test = RandomForest_tree.predict(x_test_pca)
model_evaluation(y_test, y_pred_test, x_train.shape[0], x_train.shape[1])
# Evaluacija modela na testnom skupu
mae_test = mean_absolute_error(y_test, y_pred_test)
#print("MAE na testnom skupu RandomForest regresija s PCA:", mae_test)
######################################################################################

########################################################################################

# all_cross_val_results = [
#     cross_val_results_Linear,
#     cross_val_resultsRidge,
#     cross_val_resultsLasso,
#     cross_val_resultsKNN,
#     cross_val_resultsRadnomForest,
#     cross_val_resultsLinearPCA,
#     cross_val_resultsRidgePCA,
#     cross_val_resultsLassoPCA,
#     cross_val_resultsKNNPCA,
#     cross_val_resultsRadnomForestPCA
# ]
# print(all_cross_val_results)


