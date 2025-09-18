---
title: Converting XGBoost Python Boosters to PMML
notebook: conversion/xgboost/python.ipynb
---

## Dataset

### DMatrix

The most suitable data container for data preparation work is Pandas dataframe.

Group columns by their intended type into subsets, and cast them using the `astype(dtype)` method.

For continuous columns, specify the "smallest" appropriate NumPy or Pandas' data type to reclaim memory. Nullable Pandas' data types such as `pandas.Float32DType` and `pandas.Int32Dtype` shine with mixed-type sparse datasets, because they ensure consistent representation of missing values as `pandas.NA`.

For categorical columns, specify the Pandas' categorical data type.

```python
X = X[float_cols + int_cols + categorical_cols]

X[float_cols] = X[float_cols].astype("Float32")
X[int_cols] = X[int_cols].astype("Int32")
X[categorical_cols] = X[categorical_cols].astype("category")
```

Prepare a `feature_types` helper parameter to freeze the interpretation of feature data.

```python
feature_types = ["float"] * len(float_cols) + ["int"] * len(int_cols) + ["c"] * len(categorical_cols)
```

Categorical targets are similar to categorical features.
However, to impose a custom order, specify the data type as a fully initialized `pandas.CategoricalDtype` object instead of a `"category"` shorthand.

```python
from pandas import CategoricalDtype

y = y.astype(CategoricalDtype(categories = ["zero", "one"]))
```

Combining everything into a `DMatrix` object:

```python
from xgboost import DMatrix

dmat = DMatrix(data = X, label = y.cat.codes, feature_types = feature_types, enable_categorical = True)
```

## Workflow

Training:

```python
import xgboost

booster = xgboost.train(params = {"objective" : "binary:logistic"}, dtrain = dmat)
booster.save_model("Booster.json")
```

## Export to PMML

Convert booster objects to PMML using the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package.

The `sklearn2pmml.sklearn2pmml(obj, pmml_path)` supports a small number of non-SkLearn artifacts, with LightGBM and XGBoost boosters being prime examples.

A booster object embeds a simplified data schema.
If the dataset contains categorical features, then override it with an external feature map so that the PMML converter can translate category level indices back to original values.

The `xklearn2pmml.xgboost.make_feature_map(X)` utility function can generate various types of feature maps.
The generation of ["pre-categorical" feature maps](index.md#feature-map) is activated by passing `category_to_indicator = True`.

```python
from sklearn2pmml.xgboost import make_feature_map

fmap = make_feature_map(X, category_to_indicator = True)
fmap.save("Booster.fmap.tsv")
```

Attach the feature map to the booster using the `fmap` attribute:

```python
booster.fmap = fmap
```

Converting:

```python
from sklearn2pmml import sklearn2pmml

sklearn2pmml(booster, "Booster.pmml")
```

The `sklearn2pmml` package currently does not support customizing the target field.

{{ resources_section() }}