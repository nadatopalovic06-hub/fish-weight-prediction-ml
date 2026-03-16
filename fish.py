import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score


# UČITAVANJE PODATAKA
df = pd.read_excel("Fish.xlsx", sheet_name=0)

# uklanjanje Weight = 0
df['Weight'] = df['Weight'].replace(0, np.nan)
df = df.dropna(subset=['Weight'])

features_all = ['Length1','Length2','Length3','Height','Width','Weight']


# Skaliranje za boxplot
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features_all]), columns=features_all)

# Histogram za Weight
plt.figure(figsize=(6,4))
plt.hist(df['Weight'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribucija težine riba (Weight)", fontsize=14)
plt.xlabel("Weight")
plt.ylabel("Broj riba")
plt.tight_layout()
plt.show()

# Boxplot za sve prediktore + Weight (skalirano)
plt.figure(figsize=(8,5))
sns.boxplot(data=df_scaled)
plt.title("Boxplot prediktora i Weight (skalirano)", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Scaled values")
plt.tight_layout()
plt.show()

# Deskriptivna statistika
desc = df.describe().round(1)
print("\nDeskriptivna statistika (min, max, mean, std):")
print(desc.loc[['min','max','mean','std']])

# 3. KORELACIJE

corr = df[features_all].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelaciona matrica")
plt.show()

# 4. LINEARNI MODELI

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return r2, rmse

X1D = df[['Length3']]
X2D = df[['Length3','Width']]
X3D = df[['Length3','Width','Height']]
y = df['Weight']

X1D_train, X1D_test, y_train, y_test = train_test_split(X1D, y, test_size=0.2, random_state=42)
X2D_train, X2D_test, _, _ = train_test_split(X2D, y, test_size=0.2, random_state=42)
X3D_train, X3D_test, _, _ = train_test_split(X3D, y, test_size=0.2, random_state=42)

linear_results = []

# 1D
lm1 = LinearRegression()
lm1.fit(X1D_train, y_train)
r2, rmse = evaluate(lm1, X1D_test, y_test)
linear_results.append(("Linear_1D", r2, rmse))

# 2D
lm2 = LinearRegression()
lm2.fit(X2D_train, y_train)
r2, rmse = evaluate(lm2, X2D_test, y_test)
linear_results.append(("Linear_2D", r2, rmse))

# 3D
lm3 = LinearRegression()
lm3.fit(X3D_train, y_train)
r2, rmse = evaluate(lm3, X3D_test, y_test)
linear_results.append(("Linear_3D", r2, rmse))

print("\nLINEARNI MODELI (pregled):")
print(pd.DataFrame(linear_results, columns=["Model","R2","RMSE"]).sort_values(by="R2", ascending=False))

# 5. POLINOMSKI MODELI

# 2D polinomijal (odabrani model)
poly2D = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly2D.fit(X2D_train, y_train)
r2_2d, rmse_2d = evaluate(poly2D, X2D_test, y_test)

# 3D polinomijal
poly3D = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly3D.fit(X3D_train, y_train)
r2_3d, rmse_3d = evaluate(poly3D, X3D_test, y_test)

poly_results = [("Polynomial_2D", r2_2d, rmse_2d), ("Polynomial_3D", r2_3d, rmse_3d)]
print("\nPOLINOMSKI MODELI:")
print(pd.DataFrame(poly_results, columns=["Model","R2","RMSE"]).sort_values(by="R2", ascending=False))


# 6. SCATTER 2D vs 3D polinomijal

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
preds2D = poly2D.predict(X2D_test)
plt.scatter(y_test, preds2D)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Stvarna težina")
plt.ylabel("Predikcija")
plt.title("Polynomial 2D: Length3 + Width")

plt.subplot(1,2,2)
preds3D = poly3D.predict(X3D_test)
plt.scatter(y_test, preds3D)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Stvarna težina")
plt.ylabel("Predikcija")
plt.title("Polynomial 3D: Length3 + Width + Height")
plt.tight_layout()
plt.show()
# 7. SCATTER: odabrani model (2D polinomijal)

plt.figure(figsize=(6,6))
plt.scatter(y_test, preds2D)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Stvarna težina")
plt.ylabel("Predikcija")
plt.title("Odabrani model: Polynomial 2D")
plt.tight_layout()
plt.show()


# Cross-validation
cv_2d = cross_val_score(poly2D, X2D, y, cv=5, scoring='r2')
cv_3d = cross_val_score(poly3D, X3D, y, cv=5, scoring='r2')

print("\nCROSS-VALIDATION REZULTATI (R²):")

print("\nPolynomial 2D:")
print("R2 po foldovima:", np.round(cv_2d, 2))
print("Prosek R2:", round(cv_2d.mean(), 2))
print("Std:", round(cv_2d.std(), 2))

print("\nPolynomial 3D:")
print("R2 po foldovima:", np.round(cv_3d, 2))
print("Prosek R2:", round(cv_3d.mean(), 2))
print("Std:", round(cv_3d.std(), 2))

#JEDNACINA ZA 2D REGRESIJU
# Ispis koeficijenata konacnog modela
poly = poly2D.named_steps['polynomialfeatures']
lin = poly2D.named_steps['linearregression']

feature_names = poly.get_feature_names_out(['Length3','Width'])
coefs = lin.coef_
intercept = lin.intercept_

print("\nKONACAN MODEL:")
print("Intercept:", intercept)

for name, coef in zip(feature_names, coefs):
    print(name, ":", coef)

# PREDIKCIJA NOVE RIBE

nova_riba = pd.DataFrame([[27, 4]], columns=['Length3','Width'])
predikcija = poly2D.predict(nova_riba)
print("\nPredikcija nove ribe (Polynomial 2D):", predikcija[0])
# Cross-validation za linearne modele

cv_lin_2d = cross_val_score(lm2, X2D, y, cv=5, scoring='r2')
cv_lin_3d = cross_val_score(lm3, X3D, y, cv=5, scoring='r2')








