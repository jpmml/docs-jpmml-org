---
title: Converting Scikit-Learn LightGBM pipelines to PMML
notebook: conversion/sklearn/lightgbm.ipynb
---

## Workflow

### Ordinal encoding

**General recipe**: pass continuous columns through unchanged, and encode categorical columns to integers.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

transformer = ColumnTransformer([
	("continuous", "passthrough", continuous_cols),
	("categorical", OrdinalEncoder(), categorical_cols)
])
```

This column transformer produces a homogeneous NumPy array.

Just like [XGBoost estimators](xgboost.md#ordinal-encoding), LightGBM estimators in their default configuration would assume that all columns are continuous columns, and partition them using continuous splits.

Override this behavious by passing the `categorical_feature` parameter.
The appropriate value is a list of categorical columns references, either as integer column indices or string column names.

The `categorical_feature` parameter is a `fit(X, y)` method parameter instead of a constructor parameter.
When the LightGBM estimator is embedded in a pipeline, pass the `categorical_feature` parameter (together with any helper parameters) via the `Pipeline.fit(X, y, **params)` method using the "parameter path syntax" (`<step_name>__<parameter_name> = <value>`).

Generating and passing `categorical_feature` as integer column indices:

```python
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

categorical_feature = [len(continuous_cols) + categorical_idx for categorical_idx in range(0, len(categorical_cols))]

classifier = LGBMClassifier(random_state = 42)

pipeline = Pipeline([
	("transformer", transformer),
	("classifier", classifier)
])
pipeline.fit(X, y, classifier__categorical_feature = categorical_feature)
```

Generating and passing `categorical_feature` as string column names:

```python
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

feature_name = continuous_cols + categorical_cols
categorical_feature = categorical_cols

classifier = LGBMClassifier(random_state = 42)

pipeline = Pipeline([
	("transformer", transformer),
	("classifier", classifier)
])
pipeline.fit(X, y, classifier__feature_name = feature_name, classifier__categorical_feature = categorical_feature)
```

Passing `categorical_feature` as string column names is more robust, because the `feature_name` helper parameter guarantees proper alignment with NumPy array dimensions.

Column names can be real or synthetic.
They appear in LightGBM model files, but not in the PMML representation of SkLearn LightGBM pipelines, as the PMML converter independently collects and maintains feature descriptions.

### Casting

**General recipe**: pass continuous columns through unchanged, and cast categorical columns to `pandas.CategoricalDtype`.

SkLearn lacks a dedicated casting transformer.
Some SkLearn transformers like `OrdinalEncoder` do have a `dtype` attribute, but it is meant for optimizing the numeric data type of output NumPy arrays not general data type casting.

The `sklearn2pmml` package provides the `sklearn2pmml.decoration.CategoricalDomain` decorator and the `sklearn2pmml.preprocessing.CastTransformer` transformer for this purpose.
Select the one that better suits the pipeline configuration and objectives.

The `CategoricalDomain` decorator must be positioned first in the pipeline.
It handles a wide variety of data ingestion tasks, including casting user-provided data to the desired data type, performing value validity checks, treating missing and invalid values, collecting metadata, etc.

In contrast, the `CastTransformer` transformer can appear anywhere in the pipeline.
It specializes on data type conversions only.

Casting to `CategoricalDtype` differs from ordinary casts (eg. from `int` to `float`, or vice versa), because it requires backing from the underlying data container.
Specifically, Pandas' extension data types are not interoperable with NumPy arrays or matrices.
For the casting to have the desired effect, collect the results into a Pandas' series or dataframe, and keep them so until the final estimator step of the pipeline.

The ML framework typically takes responsibility for data container management.

Most SkLearn transformers support both NumPy and Pandas data containers.
Interconversions can happen at the meta-estimator level, when collecting and aggregating child transformer outputs.
The default behaviour is downcasting Pandas' data containers to NumPy arrays, and downcasting Pandas' extension data types to NumPy `object` data type.
Override this using the `set_output` API.

Calling `set_output(transform = "pandas")` on either the child column transformer or the parent pipeline "fixes" the current workflow.

```python
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn2pmml.preprocessing import CastTransformer

transformer = ColumnTransformer(
    [("continuous", "passthrough", continuous_cols)] +
    [(categorical_col, CastTransformer("category"), [categorical_col]) for categorical_col in categorical_cols]
)
transformer.set_output(transform = "pandas")

classifier = LGBMClassifier(random_state = 42)

pipeline = Pipeline([
    ("transformer", transformer),
    ("classifier", classifier)
])
pipeline.fit(X, y)
```

The LightGBM estimator can now detect categorical columns by the `CategoricalDtype` data type.
There is no need for messing with fit params anymore.

The `lightgbm` package has been relying on the `pandas` package for categorical data support since its earliest versions.

LightGBM estimators are more robust than XGBoost estimators, because they encode the effective mapping between category levels and category indices into LightGBM model files as a `pandas_categorical` section (cf. an external feature map file).
However, the primary index on this mapping may be unstable if the training dataset contained unused features.

{{ resources_section() }}