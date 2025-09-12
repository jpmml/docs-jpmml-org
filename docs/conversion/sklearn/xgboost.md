---
title: Converting Scikit-Learn XGBoost pipelines to PMML
notebook: conversion/sklearn/xgboost.ipynb
---

## Dataset

### Features

XGBoost estimators need to be parameterized both for the training algorithm (booster) and the dataset.
The latter requirement stems from the fact that Scikit-Learn API must convert the user-provided NumPy matrix or Pandas' dataframe to XGBoost's `DMatrix` before dispatching to the Learning API.

Important dataset parameters:

* `feature_types`
* `enable_categorical`
* `missing`

Incomplete or incorrect dataset parameterization typically does not raise errors.
XGBoost will simply fall back to safe defaults, and yield a fully-functional booster, albeit with reduced predictive performance relative to the optimal one.

Such fallbacks are hard to detect due to limited tooling.
The best option is to export the booster in some text-based data format, and manually inspect the tree structure for notable structural deficiencies (eg. not seeing any categorical splits).

### Labels

XGBoost classifiers need extra attention, because their ability to accept non-numeric class labels has changed considerabily between versions.

Versions 0.4 through 1.7 adhered to SkLearn conventions that if custom class labels (eg. strings) were passed to the `fit(X, y)` method, then the `predict(X)` method yielded the same.
Unfortunately, in version 2.0 and newer, this is no longer the case.

Encoding and decoding class labels manually:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Encode from strings to integers
y_xgb = le.fit_transform(y).reshape((-1, ))

# Fit and predict integers
yt_xgb = classifier.fit_predict(X, y_xgb)

# Decode from integers back to strings
yt = le.inverse_transform(yt_xgb)
```

To persist the effective label encoder, assign it to the classifier as the `_le` attribute:

```python
classifier._le = le
```

## Pipeline

### Ordinal encoding (legacy)

Columns are separated by operational type into continuous and categorical subsets.

The continuous subset may be further separated by data type into floating-point and integer subsets.
However, this refinement is unnecessary from the training perspective, because `DMatrix` mangles all feature values into the same `float32` data type internally.

**General recipe**: pass continuous columns through unchanged, and encode categorical columns to integers.

If you want categorical splits, then the list of suitable transformers is limited to `OrdinalEncoder` (direct mapping) and `TargetEncoder` (pre-processed mapping).
Most importantly, avoid the `OneHotEncoder` transformer, because it binarizes data making advanced partitioning operations infeasible. 

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

transformer = ColumnTransformer([
	("continuous", "passthrough", continuous_cols),
	("categorical", OrdinalEncoder(), categorical_cols)
])
```

The `ColumnTransformer` meta-transformer produces a NumPy matrix by default.

XGBoost estimators cannot guess the user intent behind each column. Left to themselves, they always assume continuous numeric data, and partition it using continuous splits.

Specify NumPy matrix column types using the `feature_types` parameter.
It accepts a list of string codes -- one per column. Use `q` (shorthand for "quantitative") for continuous columns and `c` for categorical ones. Finer typing is possible, but not necessary.

Generating and passing `feature_types`:

```python
from xgboost import XGBClassifier

feature_types = ["q"] * len(continuous_cols) + ["c"] * len(categorical_cols)

classsifier = XGBClassifier(feature_types = feature_types)
```

### Casting (modern)

The main difference is converting any user-provided data container to Pandas' dataframe, which then enables casting columns into maximally specific type. 

The first part is easy. Simply instruct the `ColumnTransformer` meta-transformed to change modes using the `set_output` API.

The second part is harder, because SkLearn lacks a casting transformer.
Suggested workarounds include writing a cast function and wrapping it into the `FunctionTransformer` transformer, or developing a custom cast transformer class from scratch.
Neither option is particularly appealing due to productionization challenges.

The `sklearn2pmml` package provides a `CastTransformer` transformer that meets all the requirements.

**General recipe**: cast continuous columns to `numpy.float32`, and cast categorical columns to `pandas.CategoricalDtype`.
When dealing with complex data types or disjoint value spaces, apply the `CastTransformer` to one column at a time.

```python
from sklearn.compose import ColumnTransformer
from sklearn2pmml.preprocessing import CastTransformer

transformer = ColumnTransformer(
	# Cast continuous columns together
	[("continuous", CastTransformer(dtype = numpy.float32), continuous_cols)] +
	# Cast categorical columns one by one due to disjoint value spaces
	[(categorical_col, CastTransformer(dtype = "category"), [categorical_col]) for categorical_col in categorical_cols]
)
transformer.set_output(transform = "pandas")
```

XGBoost estimators can now collect all feature information from Pandas' dataframes automatically, making the `feature_types` parameter obsolete.

However, for as long as categorical splits carry the experimental status, users must explicitly greenlight their use by passing `enable_categorical = True`:

```python
from xgboost import XGBClassifier

classifier = XGBClassifier(enable_categorical = True)
```

{{ resources_section() }}
