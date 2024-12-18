import numpy as np
from boosting import GradientBoostingMSE
from random_forest import RandomForestMSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Generate sample data
np.random.seed(43)
X = np.random.rand(1000, 5)
X = np.abs(X)
true_weights = np.array([1.5, 2.0, 0.0, 3.0, 0.5])
y = X @ true_weights + np.abs(np.random.randn(1000)) * 0.5

# Split data/
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1,
    'random_state': 42
}
gb_custom = GradientBoostingMSE(n_estimators=100,
            learning_rate=0.1,
            tree_params={'max_depth' : 10, 'random_state': 42})
gb_sklearn = GradientBoostingRegressor(**params)

rf_custom = RandomForestMSE(n_estimators=100,
                            tree_params={'max_depth' : 10, 'random_state': 42})

rf_sklearn = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42)

# Train models
history = gb_custom.fit(X_train, y_train, trace=True)
gb_sklearn.fit(X_train, y_train)
rf_custom.fit(X_train, y_train, trace=True)
rf_sklearn.fit(X_train, y_train)

# Make predictions
y_pred_custom_gb = gb_custom.predict(X_test)
y_pred_sklearn_gb = gb_sklearn.predict(X_test)
y_pred_custom_rf = rf_custom.predict(X_test)
y_pred_sklearn_rf = rf_sklearn.predict(X_test)

# Evaluate performance
mse_custom = mean_squared_error(y_test, y_pred_custom_gb)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn_gb)
mse_custom_rf = mean_squared_error(y_test, y_pred_custom_rf)
mse_sklearn_rf = mean_squared_error(y_test, y_pred_sklearn_rf)

print(f"MSE Custom GB: {mse_custom:.4f}")
print(f"MSE Sklearn GB: {mse_sklearn:.4f}")


print(f"MSE Custom RF: {mse_custom_rf:.4f}")
print(f"MSE Sklearn RF: {mse_sklearn_rf:.4f}")




print(f"Count of trees: {len(history['train'])}")
print()
print('history GB loss:')
print(np.array(history['train']).reshape(-1, 1))
