# Инструкция

Можно импортировать 2 класса классных - **GradientBoostingMSE** и **RandomForestMSE**

```from ensembles import GradientBoostingMSE, RandomForestMSE```

## Описание классов

### RandomForestMSE

Кастомная реализация случайного леса для регрессии с использованием `sklearn's` `DecisionTreeRegressor`.

#### Параметры:
- `n_estimators` (int): Количество деревьев в лесу.
- `tree_params` (dict[str, Any], optional): Параметры для `DecisionTreeRegressor`.

#### Методы:
- `fit(X, y, X_val=None, y_val=None, trace=None, patience=None)`: Обучает ансамбль деревьев.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of objects, array of shape (n_val_objects, n_features). Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set of labels, array of shape (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate RMSLE while training. True by default if validation data is provided. Defaults to None.
            patience (int | None, optional): Number of training steps without decreasing the train loss (or validation if provided), after which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory` if `trace=True` or if validation data is provided.

- `predict(X)`: Делает предсказания, усредняя результаты всех деревьев.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).

### GradientBoostingMSE

Кастомная реализация градиентного бустинга для регрессионных задач.

#### Параметры:
- `n_estimators` (int): Количество итераций бустинга.
- `learning_rate` (float): Управляет вкладом каждого дерева.
- `tree_params` (dict[str, Any]): Параметры для `DecisionTreeRegressor`.

#### Методы:
- `fit(X, y, X_val=None, y_val=None, trace=None, patience=None)`: Обучает модель градиентного бустинга.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of objects, array of shape (n_val_objects, n_features). Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set of labels, array of shape (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate rmsle while training. True by default if validation data is provided. Defaults to None.
            patience (int | None, optional): Number of training steps without decreasing the train loss (or validation if provided), after which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory` if `trace=True` or if validation data is provided.rams (dict[str, Any] | None, optional): Parameters for sklearn trees. Defaults to None.

- `predict(X)`: Делает предсказания, суммируя вклад всех деревьев.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).