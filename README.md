# Fish Weight Prediction
This project uses machine learning regression models to predict fish weight based on physical measurements.

## Dataset
The dataset contains measurements of different fish species, including:

- Length1
- Length2
- Length3
- Height
- Width
- Weight

Rows with Weight = 0 were removed before analysis.

## Data Analysis
The project includes:

- Histogram of fish weight distribution
- Boxplot visualization of features
- Correlation heatmap
- Descriptive statistics

## Models
The following regression models were tested:

### Linear Regression
- 1D model (Length3)
- 2D model (Length3 + Width)
- 3D model (Length3 + Width + Height)

### Polynomial Regression
- Polynomial 2D
- Polynomial 3D

## Evaluation
Models were evaluated using:

- R² score
- RMSE
- 5-fold cross validation

## Final Model
The best performing model was **Polynomial Regression (2D)** which was using:
Length3 and Width as predictors.

The model can also predict the weight of a new fish based on these measurements.

## Technologies Used
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
