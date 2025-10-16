# Model Training Component

A flexible and extensible model training component that follows the scikit-learn API while supporting various machine learning libraries including scikit-learn, XGBoost, CatBoost, and custom models.

## Features

- **Scikit-learn API Compatibility**: Fully compatible with scikit-learn pipelines and utilities
- **Multi-Library Support**: Works with scikit-learn, XGBoost, CatBoost, and custom models
- **Extensible Design**: Easy to add custom models while maintaining compatibility
- **Pipeline Integration**: Seamless integration with scikit-learn pipelines
- **Best Practices**: Follows Python and ML best practices with comprehensive error handling

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from frigg_ml.src.model_training import ModelTrainer

# Create and train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
trainer = ModelTrainer(model)
trainer.fit(X_train, y_train)

# Make predictions
predictions = trainer.predict(X_test)
probabilities = trainer.predict_proba(X_test)  # For classification models
```

## Supported Models

### Scikit-learn Models
All scikit-learn estimators are supported out of the box:

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# Classification
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
trainer = ModelTrainer(rf_clf)

# Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
trainer = ModelTrainer(rf_reg)
```

### XGBoost Models
```python
from xgboost import XGBClassifier, XGBRegressor

# Classification
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
trainer = ModelTrainer(xgb_clf)

# Regression
xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
trainer = ModelTrainer(xgb_reg)
```

### CatBoost Models
```python
from catboost import CatBoostClassifier, CatBoostRegressor

# Classification
cat_clf = CatBoostClassifier(iterations=100, random_state=42, verbose=False)
trainer = ModelTrainer(cat_clf)

# Regression
cat_reg = CatBoostRegressor(iterations=100, random_state=42, verbose=False)
trainer = ModelTrainer(cat_reg)
```

## Creating Custom Models

### Option 1: Inherit from BaseCustomModel

```python
from frigg_ml.src.model_training import BaseCustomModel
import numpy as np

class MyCustomModel(BaseCustomModel):
    def __init__(self, param1=1.0, param2=100):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted_ = False
    
    def fit(self, X, y, **kwargs):
        # Your training logic here
        self.weights_ = np.random.rand(X.shape[1])
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        return X.dot(self.weights_)

# Usage
custom_model = MyCustomModel(param1=0.5, param2=200)
trainer = ModelTrainer(custom_model)
```

### Option 2: Inherit from SklearnCompatibleModel

For full scikit-learn compatibility (recommended):

```python
from frigg_ml.src.model_training import SklearnCompatibleModel
import numpy as np

class MySklearnModel(SklearnCompatibleModel):
    def __init__(self, param1=1.0, param2=100):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted_ = False
    
    def fit(self, X, y, **kwargs):
        # Your training logic here
        self.weights_ = np.random.rand(X.shape[1])
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        return X.dot(self.weights_)

# This model will have get_params() and set_params() methods automatically
model = MySklearnModel(param1=0.5)
params = model.get_params()  # Returns {'param1': 0.5, 'param2': 100}
model.set_params(param1=0.8)  # Updates param1 to 0.8
```

### Option 3: Any Class with fit() and predict() Methods

```python
class SimpleModel:
    def __init__(self):
        self.is_fitted_ = False
    
    def fit(self, X, y):
        # Simple mean prediction
        self.mean_ = np.mean(y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        return np.full(len(X), self.mean_)

# Works directly with ModelTrainer
simple_model = SimpleModel()
trainer = ModelTrainer(simple_model)
```

## Advanced Usage

### Setting Additional Model Parameters

```python
from sklearn.ensemble import RandomForestClassifier

# Create model with basic parameters
model = RandomForestClassifier(random_state=42)

# Set additional parameters through ModelTrainer
additional_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5
}

trainer = ModelTrainer(model, model_params=additional_params)
# Model now has n_estimators=200, max_depth=10, min_samples_split=5
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Create a pipeline
model = RandomForestClassifier(random_state=42)
trainer = ModelTrainer(model)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', trainer)
])

# Use like any sklearn estimator
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(random_state=42)
trainer = ModelTrainer(model)

# Define parameter grid (note the 'model__' prefix)
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(trainer, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_trainer = grid_search.best_estimator_
```

## API Reference

### ModelTrainer Class

#### Constructor
```python
ModelTrainer(model, model_params=None)
```

**Parameters:**
- `model`: Any model instance with `fit()` and `predict()` methods
- `model_params` (optional): Dictionary of additional parameters to set on the model

#### Methods

##### fit(X, y, **fit_params)
Fit the model to training data.

**Parameters:**
- `X`: Training features (array-like or DataFrame)
- `y`: Training targets (array-like or Series)  
- `**fit_params`: Additional parameters passed to the model's fit method

**Returns:** self (for method chaining)

##### predict(X)
Make predictions using the fitted model.

**Parameters:**
- `X`: Input features (array-like or DataFrame)

**Returns:** Predicted values (numpy array)

##### predict_proba(X)
Predict class probabilities (classification models only).

**Parameters:**
- `X`: Input features (array-like or DataFrame)

**Returns:** Predicted probabilities (numpy array)

#### Properties

##### feature_importances_
Get feature importances if available from the underlying model.

**Returns:** Feature importances (numpy array) or None

#### Attributes

- `model_`: The fitted model instance
- `is_fitted_`: Whether the model has been fitted
- `n_features_in_`: Number of features seen during fit
- `feature_names_in_`: Names of features (if input was DataFrame)

## Examples

See the `examples.py` file for comprehensive examples demonstrating:

- Basic usage with various model types
- Pipeline integration
- Custom model implementation
- Parameter configuration
- Cross-validation and hyperparameter tuning

To run the examples:

```python
from frigg_ml.src.model_training.examples import run_all_examples
run_all_examples()
```

## Best Practices

1. **Always validate your custom models** by testing with the ModelTrainer before using in production
2. **Use the SklearnCompatibleModel base class** for custom models when possible
3. **Handle errors gracefully** in your custom model's fit() and predict() methods
4. **Document your custom models** clearly, including parameter descriptions
5. **Test compatibility** with scikit-learn pipelines and utilities
6. **Use type hints** in your custom model implementations

## Error Handling

The ModelTrainer provides comprehensive error handling:

- **Validation errors**: When models don't have required methods
- **Parameter errors**: When invalid parameters are provided
- **Fitting errors**: When model training fails
- **Prediction errors**: When prediction fails
- **Feature consistency**: When prediction data doesn't match training data

All errors include clear, descriptive messages to help with debugging.

## Testing

Run the test suite to verify everything works correctly:

```bash
# Run all tests
pytest tests/model_training/

# Run specific test file
pytest tests/model_training/test_model_trainer.py

# Run with coverage
pytest tests/model_training/ --cov=frigg_ml.src.model_training
```
