---
title: Converting Scikit-Learn NGBoost pipelines to PMML
notebook: conversion/sklearn/ngboost.ipynb
---

## Overview

NGBoost learns (approximates) the distribution function of target values, rather than individual point estimates.

Knowing the parametric form of a distribution function opens up unique opportunities.
For example, predicting quantiles (eg. "predict the target along with its q5 and q95 quantiles") or prediction intervals (eg. "predict the target with a range that includes the true value with 90% confidence").

Classical Scikit-Learn estimators can achieve similar objectives.
For example, predict quantiles using `QuantileRegressor` or `GradientBoostingRegressor` (with the pinball loss), then wrap the elementary models into a single deployable unit using `MultiOutputRegressor`.

However, NGBoost outperforms such setups in many ways.
It produces smaller and computationally more efficient models and, most importantly, its predictions are heteroskedastic rather than homoskedastic.

To elaborate, NGBoost does not predict the target value for a sample directly.
It predicts the parameters of the distribution function for a sample, constructs the corresponding distribution instance, and evaluates it for the mean (q50).

All samples share the same distribution family, but they each parameterize it differently.
Specifically, a sample's confidence is reflected in its estimated dispersion parameter value(s).
Samples from higher-density (ie. more typical) regions have lower dispersion, which naturally leads to narrower prediction intervals.

## Workflow

NGBoost estimators follow Scikit-Learn API conventions, making them usable in standalone mode as well as when embedded in pipelines.
Adopt the pipeline approach proactively, because real-life datasets almost always require some form of transformation before they can be passed to the final estimator.

Data pre-processing requirements depend on the NGBoost base estimator (the `Base` parameter).
The default base estimator is `sklearn.tree.DecisionTreeRegressor`, which accepts continuous features as-is, but requires categorical features to be encoded:

```python
from ngboost.learners import default_tree_learner
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output = False), cat_cols),
    ("cont", "passthrough", cont_cols)
])

base_estimator = default_tree_learner
```

Adjust the data pre-processor and base estimator in tandem when tackling more complex problems.
For example, if the dataset is sparse and contains many high-cardinality categorical features, then consider using `xgboost.XGBRegressor` as the base estimator:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

transformer = ColumnTransformer([
    ("cat", OrdinalEncoder(), cat_cols),
    ("cont", "passthrough", cont_cols)
])

feature_types = ["c"] * len(cat_cols) + ["q"] * len(cont_cols)

base_estimator = XGBRegressor(n_estimators = 3, max_depth = 2, feature_types = feature_types)
```

Some words of caution and advice.

First, prefer the most basic and backwards-compatible solution whenever several technical approaches are feasible.
For example, `XGBRegressor` accepts categorical features either in ordinally-encoded or `pandas.CategoricalDtype` forms.
At the time of writing this, NGBoost only supports the former.
This limitation stems from the fact that NGBoost casts all incoming data containers to NumPy arrays, which discards or disfigures any specialized third-party data representations (eg. Pandas' categoricals).

Second, NGBoost estimator classes deviate from the core Scikit-Learn convention of storing fitted state in attributes ending with the underscore character (`_`).
Scikit-Learn 1.8.0 started checking the fitted state of pipeline steps during prediction.
This change causes all NGBoost pipelines to fail with a `sklearn.exceptions.NotFittedError`, whether actually justified or not.

As a quick fix, set an indicator attribute:

```python
pipeline = ...
pipeline.fit(X, y)

# Mark the regressor as fitted
pipeline["regressor"].fitted_ = True
```

Alternatively, exclude the NGBoost estimator from the pipeline, and emulate the transform-then-predict sequence manually.
Apply the same pattern when calling the non-standard `NGBoost.pred_dist(X)`  method:

```python
pipeline = ...
pipeline.fit(X, y)

# Steps up until the final step
Xt = pipeline[:-1].transform(X)

# The final step, with a non-standard predict method
y_dist = pipeline._final_estimator.pred_dist(Xt)
```

## Export to PMML

Convert NGBoost artifacts to PMML as usual:

```python
from sklearn2pmml import sklearn2pmml

sklearn2pmml(pipeline, "NGBoostPipeline.pmml")
```

By default, the converter generates PMML that replicates the point-prediction behaviour of `predict(X)` and `predict_proba(X)` methods.
Activate the generation of NGBoost-specific prediction aspects on a per-aspect basis.

Extending the model schema with more output fields is easy.
However, do not get carried away.
Adding unnecessary or undocumented fields is likely a net negative, because it increases complexity for model consumers.

Supported configurations:

| Estimator | Distribution | Parameters | Prediction intervals |
|-----------|--------------|------------|----------------------|
| `NGBRegressor` | `Normal` | `loc`, `scale` | Yes |
| `NGBRegressor` | `LogNormal` | `loc`, `scale` | Yes |
| `NGBRegressor` | `Poisson` | `loc` | No |
| `NGBClassifier` | `Bernoulli` | `loc` | No |
| `NGBClassifier` | `k_categorical` | `loc(0)`, `loc(1)`, ..., `loc(<k-1>)` | No |

### Distribution parameters

The predicted parameter values are available internally under pre-defined names.

For example, consider the scoring of log-normal distribution models.
The first two child models predict the `loc` and `scale` parameter values.
The last child model retrieves them by name, and calculates the log-normal mean of the target value as `exp(loc + exp(2.0 * scale) / 2.0)`.

If the application wants to reconstruct and perform custom work with the log-normal distribution instance on its side, then it needs the same `loc` and `scale` parameter values.

When exposing internal fields, give the final `OutputField` elements more meaningful names, and apply relevant transformations.
This spares application developers from having to re-discover and re-implement the underlying statistics.
For example, expose the `loc` parameter directly as `dist(mu)`, but exponentiate the `scale` parameter before exposing it as `dist(sigma)`.

The SkLearn2PMML post-processing mechanism is the easiest way to generate and append custom output fields straight to the final model element, especially when working with complex model ensembles:

```python
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.cross_reference import Recaller
from sklearn2pmml.decoration import Alias
from sklearn2pmml.preprocessing import ExpressionTransformer    

def make_branch(in_name, expr, out_name):
    recaller = Recaller(memory = None, names = [in_name])
    transformer = Alias(ExpressionTransformer(expr), name = out_name, prefit = True)
    return make_pipeline(recaller, transformer)

dist_output = FeatureUnion([
    ("loc", make_branch("loc", "X['loc'] + 0.0", "dist(mu)")),
    ("scale", make_branch("scale", "numpy.exp(X['scale'])", "dist(sigma)"))
])

pipeline.predict_transformer = dist_output

sklearn2pmml(pipeline, "NGBoostPipeline-dist.pmml")
```

### Prediction intervals

For NGBoost regressors, the most common use of the `scale` parameter is for calculating prediction intervals.

The SkLearn2PMML package supports the `confidence_level` conversion option, which generates a pair of output fields (upper and lower bound) in one go.

Set conversion options using the `PMMLPipeline.configure(**pmml_options)` convenience method:

```python
pipeline = PMMLPipeline(...)
pipeline.fit(X, y)
pipeline.configure(confidence_level = ...)
```

Conversion options must be set after fitting, never before.
The reason is that Scikit-Learn pipelines clone the steps on each fit, and the default cloning algorithm discards any custom attributes, including PMML conversion options.

Strictly speaking, the `confidence_level` conversion option relates to total coverage (probability) via a symmetric two-sided prediction interval.
For example, `confidence_level = 0.90` represents a prediction interval from the 5th to the 95th percentile (90% coverage, 5% tails on each side).

Translation to quantiles:

```python
confidence_level = 0.90

lower_quantile = (1 - confidence_level) / 2 		# 0.05
upper_quantile = 1 - (1 - confidence_level) / 2		# 0.95
```

The converter handles `confidence_level` values as follows:

| Value | Bounds | Input schema | Output schema |
|-------|--------|--------------|---------------|
| boolean `False` | None | None | None |
| float `0.0 < x < 1.0` | Static | None | `lower(<target>)` and `upper(<target>)` |
| boolean `True` | Dynamic | Optional `ci` field | Optional `lower(<target>)` and `upper(<target>)` |
| string | Dynamic | Optional user-named field | Optional `lower(<target>)` and `upper(<target>)` |

#### No bounds

Set `confidence_level` to the `False` boolean value:

```python
pipeline.configure(confidence_level = False)
```

The converter does not generate any extra output fields.

The `scale` parameter is functionally redundant in normal distribution models.
If the application does not need it, then delete the corresponding component from the model chain.
Expect the size of the PMML document to be cut in half and, consequently, its scoring throughput to double.

#### Static bounds

Set `confidence_level` to a float value between `0` and `1` (probability):

```python
pipeline.configure(confidence_level = 0.90)

sklearn2pmml(pipeline, "NGBoostPipeline-static.pmml")
```

The converter hard-codes this value into critical z-value calculations.
The model always outputs valid `lower(<target>)` and `upper(<target>)` values.

#### Dynamic bounds

Set `confidence_level` to the `True` boolean value, or a non-empty string value.
In fact, setting `True` is equivalent to setting `"ci"`:

```python
pipeline.configure(confidence_level = "ci")

sklearn2pmml(pipeline, "NGBoostPipeline-dynamic.pmml")
```

The converter generates a `DataField` element with that name, and references it in all critical z-value calculations.
The bounds respond dynamically to the confidence level value in model scoring arguments.
The model outputs missing `lower(<target>)` and `upper(<target>)` values if the input is missing.

Scoring with dynamic bounds:

```python
from jpmml_evaluator import make_evaluator

evaluator = make_evaluator("NGBoostPipeline-dynamic.pmml") \
	.verify()

pmml_X = X.copy()
# Same coverage for all samples
pmml_X["ci"] = 0.90

pmml_y = evaluator.evaluateAll(pmml_X)
print(pmml_y)
```

Applications that need even more dynamism should fall back to exporting distribution parameters.
This allows reconstructing distribution instances locally, with full flexibility for querying and calculations.

{{ resources_section() }}