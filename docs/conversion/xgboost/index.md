---
title: Converting XGBoost to PMML
---

## Overview

XGBoost characteristics:

* Uses custom data container. A workflow must convert the user-provided data container to XGBoost's `DMatrix` before dispatching to XGBoost's train or predict methods. Leads to elevated memory consumption compared to other ML frameworks, because two copies of the same data exist simultaneously.
* Uses `float32` data type for data representation and computation. A workflow may perform "narrowing" type casts (eg. from `float64` to `float32` or `int`) to reclaim memory.
* Treats continuous and categorical features using different splitting algorithms. A workflow should pre-process categorical columns in a specific way, and activate a "categorical enabled" mode during training.
* Treats valid (ie. non-missing) and missing values using different splitting algorithms. A workflow may choose to impute missing values, or keep them as-is. The training shall yield a different model for either case.

## Dataset

### DMatrix

A `DMatrix` holds both feature and label data.
It can be parameterized directly with most common data containers such as NumPy arrays, Pandas' dataframes or R matrices.

All values are munged to `float32` data type internally.

**Feature data** is interpreted according to helper parameters `enable_categorical`, `feature_types` and `missing`.

The `feature_types` parameter determines which splitting type is appropriate for each column.
Obviously, continuous columns should be partitioned using continuous splits, and categorical columns using categorical splits.
Messing up the two may have only a limited impact on model's predictive performance, but completely undermines its interpretability.

The distinction between floating-point and integer columns is virtual.
They are partitioned using the same continuous splits. In one case the split threshold value is rendered as-is, whereas in the other case it is rounded.

**Label data** is interpreted by the objective function.

Classification tasks require users to encode complex class labels (eg. strings) into integers.
Binary classification assumes a specific order, where the inactive/no-event class label (ie. `0`) comes first and the active/event class label (ie. `1`) comes second.
Multi-class classification assumes any order.

### Feature map

A feature map organizes feature descriptions into a table-like data structure.

Each entry consists of three fields:

1. Index. A 0-based index indicating the column index in the training dataset.
2. Name. A string indicating the column name in the training dataset. For binary indicator features, the name follows the pattern `<column name>=<value>`.
3. Type code. A string enum, either `q` (quantitative), `i` (binary indicator) or `c` (categorical). The `q` type code may be replaced with `float` (floating-point number) or `int` (integer number) type codes for extra detail.

Entries are ordered by index, and stored each entry on a line by itself, using the tab character (`\t`) as field separator.
This layout is identical to the headerless tab-separated values (TSV) data format.

Feature maps have lost relevance since XGBoost switched to a JSON-based model data format.
The booster object now embeds the same information directly using its `feature_names` and `feature_types` attributes, enabling automated data schema checks during prediction.

Unfortunately, XGBoost still cannot help with the encoding of categorical features.
The booster object refers to category levels by their indices, but does not store the actual mapping itself.
Users must find their own ways for working around this deficiency.

The JPMML-XGBoost library relies on "pre-categorical" feature maps, where categorical features are represented as lists of binary indicator features. The position of each category level in the list indicates its index.

For example, encoding traffic light colors in order `[green, yellow, red]`:

```text
0	color=green	i
1	color=yellow	i
2	color=red	i
```

## Export to PMML

### XGBoost

When faced with an unknown booster object in a non-JSON data format (proprietary binary or UBJSON), then start by converting it to the JSON data format for better approachability:

```python
from xgboost import Booster

booster = Booster()
# Read binary data format
booster.load_model("Booster.bin")

# Write JSON data format
booster.save_model("Booster.json")
```

Inspect the booster-level version attribute, and learner-level data schema attributes:

```python
import json

with open("Booster.json", "r") as js_file:
	jsbooster = json.load(js_file)

print(jsbooster["version"])

learner = jsbooster["learner"]

print(learner["feature_names"])
print(learner["feature_types"])
```

The data schema should be complete and fully recognizable.
Address any issues by (re-)setting learner-level `feature_names` and `feature_types` attributes (eg. changing the name or type code of a feature), or by preparing an external feature map.

### JPMML-XGBoost

Convert XGBooost model files to PMML using the [JPMML-XGBoost](https://github.com/jpmml/jpmml-xgboost) command-line application.

Required arguments:

* `--model-input`. Path to the input XGBoost model file.
* `--pmml-output`. Path to the output PMML file.

Optional arguments:

* `--fmap-input`. Path to the input feature map file.
* `--target-name`. The name of the target field. Defaults to `_target`.
* `--target-categories`. The comma-separated list of category levels of the target field. Defaults to 0-based indices `[0, 1, ..., n_classes - 1]`.

For example, converting a binary classifier with a custom target field:

```bash
java -jar pmml-xgboost-example-executable-${version}.jar --model-input Booster.json --fmap-input Booster.fmap.tsv --pmml-output Booster.pmml --target-name "event" --target-categories "no,yes"
```
