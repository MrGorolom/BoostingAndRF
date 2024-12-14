import numpy as np
from boosting import GradientBoostingMSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)
y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(1000)

# Split data
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

# Train models
history = gb_custom.fit(X_train, y_train, trace=True)
gb_sklearn.fit(X_train, y_train)

# Make predictions
y_pred_custom = gb_custom.predict(X_test)
y_pred_sklearn = gb_sklearn.predict(X_test)

# Evaluate performance
mse_custom = mean_squared_error(y_test, y_pred_custom)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

print(f"MSE Custom: {mse_custom:.4f}")
print(f"MSE Sklearn: {mse_sklearn:.4f}")
print(f"Count of trees: {len(history['train'])}")
print()
print('history loss:')
print(np.array(history['train']).reshape(-1, 1))
