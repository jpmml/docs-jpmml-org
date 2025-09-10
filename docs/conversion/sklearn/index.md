---
title: Converting Scikit-Learn to PMML
notebook: conversion/sklearn/index.ipynb
---

Scikit-Learn (SkLearn) is supports many data science styles.
On one end, there are free-form scripts that use SkLearn utility functions sparingly.
On the other end, there are pipelines, which pack SkLearn transformers and the final SkLearn model into an integral, easily commandable (ie. fit, predict) and shareable entity.

PMML is capable of supporting both styles.
However, from the conversion perspective, the pipeline style is preferable to functional styles, because it enables easy automation.

A typical PMML-oriented SkLearn pipeline has the following stages:

* Feature mapping (E). Select and order raw features.
* Feature transformation (T). Transform features to improve their statistical properties, derive new features based on existing features.
* Feature selection (L). Filter transformed features.
* Model. Estimate the mathematical relationship between the selected features and the target.

For example:

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Combined feature mapper and transformer
transformer = ColumnTransformer([
	("continuous", StandardScaler(), continuous_cols),
	("categorical", OneHotEncoder(), categorical_cols)
], remainder = "drop")

# Feature selector
selector = SelectKBest(k = ...)

# Model
classifier = LogisticRegression(random_state = 42)

pipeline = Pipeline([
	("transformer", transformer),
	("selector", selector),
	("classifier", classifier)
])
pipeline.fit(X, y)
```

SkLearn artifacts can be converted to PMML using the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package.

This package provides two main interaction points.

The `sklearn2pmml.sklearn2pmml(obj, pmml_path)` utility function is applicable to live objects:

```python
from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "Pipeline.pmml")
```

The `sklearn2pmml` command-line application is applicable to previously serialized (in various flavours of the Pickle data format) objects:

```bash
sklearn2pmml -i Pipeline.pkl -o Pipeline.pmml
```

{{ resources_section() }}
