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

# 1. DATA LOADING AND CLEANING

# Load dataset from Excel file
df = pd.read_excel("Fish.xlsx", sheet_name=0)

# Replace Weight = 0 with NaN and remove those rows
df['Weight'] = df['Weight'].replace(0, np.nan)
df = df.dropna(subset=['Weight'])

# List of numerical features used in analysis
features_all = ['Length1','Length2','Length3','Height','Width','Weight']


# 2. DATA VISUALIZATION

# Scale data for boxplot visualization
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features_all]), columns=features_all)

# Histogram of fish weight distribution
plt.figure(figsize=(6,4))
plt.hist(df['Weight'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Fish Weight", fontsize=14)
plt.xlabel("Weight")
plt.ylabel("Number of fish")
plt.tight_layout()
plt.show()

# Boxplot for all predictors and target variable (scaled)
plt.figure(figsize=(8,5))
sns.boxplot(data=df_scaled)
plt.title("Boxplot of Features (Scaled)", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Scaled values")
plt.tight_layout()
plt.show()


# 3. DESCRIPTIVE STATISTICS
# Basic descriptive statistics of the dataset
desc = df.describe().round(1)
print("\nDescriptive statistics (min, max, mean, std):")
print(desc.loc[['min','max','mean','std']])


# 4. CORRELATION ANALYSIS

# Compute correlation matrix between variables
corr = df[features_all].corr()

# Visualize correlations using a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# 5. LINEAR REGRESSION MODELS


# Function to evaluate regression model performance
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return r2, rmse

# Different feature combinations
X1D = df[['Length3']]
X2D = df[['Length3','Width']]
X3D = df[['Length3','Width','Height']]
y = df['Weight']

# Train/test split
X1D_train, X1D_test, y_train, y_test = train_test_split(X1D, y, test_size=0.2, random_state=42)
X2D_train, X2D_test, _, _ = train_test_split(X2D, y, test_size=0.2, random_state=42)
X3D_train, X3D_test, _, _ = train_test_split(X3D, y, test_size=0.2, random_state=42)

linear_results = []

# 1D Linear Regression
lm1 = LinearRegression()
lm1.fit(X1D_train, y_train)
r2, rmse = evaluate(lm1, X1D_test, y_test)
linear_results.append(("Linear_1D", r2, rmse))

# 2D Linear Regression
lm2 = LinearRegression()
lm2.fit(X2D_train, y_train)
r2, rmse = evaluate(lm2, X2D_test, y_test)
linear_results.append(("Linear_2D", r2, rmse))

# 3D Linear Regression
lm3 = LinearRegression()
lm3.fit(X3D_train, y_train)
r2, rmse = evaluate(lm3, X3D_test, y_test)
linear_results.append(("Linear_3D", r2, rmse))

# Display comparison of linear models
print("\nLINEAR MODELS:")
print(pd.DataFrame(linear_results, columns=["Model","R2","RMSE"]).sort_values(by="R2", ascending=False))


# 6. POLYNOMIAL REGRESSION MODELS


# Polynomial regression with two predictors
poly2D = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly2D.fit(X2D_train, y_train)
r2_2d, rmse_2d = evaluate(poly2D, X2D_test, y_test)

# Polynomial regression with three predictors
poly3D = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly3D.fit(X3D_train, y_train)
r2_3d, rmse_3d = evaluate(poly3D, X3D_test, y_test)

# Model comparison
poly_results = [("Polynomial_2D", r2_2d, rmse_2d), ("Polynomial_3D", r2_3d, rmse_3d)]
print("\nPOLYNOMIAL MODELS:")
print(pd.DataFrame(poly_results, columns=["Model","R2","RMSE"]).sort_values(by="R2", ascending=False))


# 7. MODEL VISUALIZATION

# Compare predictions vs true values for polynomial models
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
preds2D = poly2D.predict(X2D_test)
plt.scatter(y_test, preds2D)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Polynomial 2D: Length3 + Width")

plt.subplot(1,2,2)
preds3D = poly3D.predict(X3D_test)
plt.scatter(y_test, preds3D)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Polynomial 3D: Length3 + Width + Height")

plt.tight_layout()
plt.show()


# 8. FINAL MODEL VISUALIZATION

# Visualization of the selected model (Polynomial 2D)
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds2D)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title("Final Model: Polynomial 2D")
plt.tight_layout()
plt.show()


# 9. CROSS VALIDATION

# Perform 5-fold cross validation
cv_2d = cross_val_score(poly2D, X2D, y, cv=5, scoring='r2')
cv_3d = cross_val_score(poly3D, X3D, y, cv=5, scoring='r2')

print("\nCROSS VALIDATION RESULTS (R²):")

print("\nPolynomial 2D:")
print("R2 per fold:", np.round(cv_2d, 2))
print("Mean R2:", round(cv_2d.mean(), 2))
print("Std:", round(cv_2d.std(), 2))

print("\nPolynomial 3D:")
print("R2 per fold:", np.round(cv_3d, 2))
print("Mean R2:", round(cv_3d.mean(), 2))
print("Std:", round(cv_3d.std(), 2))

# 10. FINAL MODEL EQUATION

# Extract polynomial feature names and regression coefficients
poly = poly2D.named_steps['polynomialfeatures']
lin = poly2D.named_steps['linearregression']

feature_names = poly.get_feature_names_out(['Length3','Width'])
coefs = lin.coef_
intercept = lin.intercept_

print("\nFINAL MODEL:")
print("Intercept:", intercept)

for name, coef in zip(feature_names, coefs):
    print(name, ":", coef)

# 11. NEW FISH PREDICTION

# Example: predict weight for a new fish
nova_riba = pd.DataFrame([[27, 4]], columns=['Length3','Width'])
predikcija = poly2D.predict(nova_riba)

print("\nPrediction for a new fish:", predikcija[0])

# 12. CROSS VALIDATION FOR LINEAR MODELS

cv_lin_2d = cross_val_score(lm2, X2D, y, cv=5, scoring='r2')
cv_lin_3d = cross_val_score(lm3, X3D, y, cv=5, scoring='r2')








