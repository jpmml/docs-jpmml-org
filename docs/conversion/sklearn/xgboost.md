---
title: Converting Scikit-Learn XGBoost pipelines to PMML
notebook: conversion/sklearn/xgboost.ipynb
---

## Dataset

### Labels

XGBoost classifiers show inconsistent support for complex class labels across major versions.

XGBoost versions 0.X through 1.X follow SkLearn conventions, allowing complex class labels as input to the `fit(X, y)` method and returning them from the `predict(X)` method.
However, XGBoost version 2.0 removed this internal label encoding logic, and newer versions have not restored it.

Use the `LabelEncoder` transformer for manual label encoding and decoding.
Be aware that it sorts strings labels in ascending lexicographic order, which may lead to conflicts during binary classification that assumes strict "no-event followed by event" order.
For example, string labels `["0", "1"]` sort correctly, but `["one", "zero"]` do not.

Customize the order by (re-)setting the `classes_` attribute:

```python
from sklearn.preprocessing import LabelEncoder

import numpy

le = LabelEncoder()
le.classes_ = numpy.asarray(["zero", "one"])

# Encode from strings to integers
# Avoid calling `fit` or `fit_transform` as it would override the custom order
y_xgb = le.transform(y).reshape((-1, ))

# Fit and predict integers
yt_xgb = classifier.fit_predict(X, y_xgb)

# Decode from integers back to strings
yt = le.inverse_transform(yt_xgb)
```

Attach the label encoder to the classifier using the `_le` attribute:

```python
classifier._le = le
```

## Workflow

### Ordinal encoding

Group columns by operational type into continuous and categorical subsets.

The continuous subset may be further grouped by data type into floating-point and integer subsets.
However, this refinement is unnecessary from the training perspective, because `DMatrix` munges all feature data into `float32` values internally.

**General recipe**: pass continuous columns through unchanged, and encode categorical columns to integers.

If you desire categorical splits, then the list of suitable transformers is limited to `OrdinalEncoder` (direct mapping) and `TargetEncoder` (pre-processed mapping).
Most importantly, avoid the `OneHotEncoder` transformer, because it binarizes data making advanced partitioning operations infeasible. 

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

transformer = ColumnTransformer([
	("cont", "passthrough", cont_cols),
	("cat", OrdinalEncoder(), cat_cols)
])
```

This column transformer produces a homogeneous NumPy array.

Just like [LightGBM estimators](lightgbm.md#ordinal-encoding), XGBoost estimators in their default configuraion would assume that all columns are continuous columns, and partition them using continuous splits.

Override this behaviour using the `feature_types` parameter.
The appropriate value is a list of string enums, where `q` denotes continuous columns and `c` categorical ones.

Generating and passing `feature_types`:

```python
from xgboost import XGBClassifier

feature_types = ["q"] * len(cont_cols) + ["c"] * len(cat_cols)

classsifier = XGBClassifier(feature_types = feature_types)
```

### Casting

The main difference is converting any user-provided data container to Pandas' dataframe, which then enables casting columns into maximally specific type. 

The first part is easy. Simply instruct the `ColumnTransformer` meta-transformer to change modes using the `set_output` API.

The second part is harder, because SkLearn lacks a casting transformer.
Suggested workarounds include writing a cast function and wrapping it into the `FunctionTransformer` transformer, or developing a custom cast transformer class from scratch.
Neither option is particularly appealing due to productionization challenges.

The `sklearn2pmml` package provides the `sklearn2pmml.preprocessing.CastTransformer` transformer that meets all the requirements.

**General recipe**: cast continuous columns to `numpy.float32`, and cast categorical columns to `pandas.CategoricalDtype`.
When dealing with complex data types or disjoint value spaces, apply the `CastTransformer` to one column at a time.

```python
from sklearn.compose import ColumnTransformer
from sklearn2pmml.preprocessing import CastTransformer

transformer = ColumnTransformer(
	# Cast continuous columns together
	[("cont", CastTransformer(dtype = numpy.float32), cont_cols)] +
	# Cast categorical columns one by one due to disjoint value spaces
	[(cat_col, CastTransformer(dtype = "category"), [cat_col]) for cat_col in cat_cols]
)
transformer.set_output(transform = "pandas")
```

XGBoost estimators can now collect all feature descriptions from Pandas' dataframes automatically, making the `feature_types` parameter obsolete.
Instead, you are required to confirm your desire for categorical splits by passing `enable_categorical = True`:

```python
from xgboost import XGBClassifier

classifier = XGBClassifier(enable_categorical = True)
```

{{ resources_section() }}
