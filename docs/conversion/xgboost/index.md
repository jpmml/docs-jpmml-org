---
title: Converting XGBoost to PMML
---

## Overview

XGBoost characteristics:

* Uses `float32` data type internally. A workflow may perform "narrowing" type casts (eg. from `float64` to `float32` or `int`) in order to preserve memory.
* Treats continuous and categorical features using different splitting algorithms. A workflow should pre-process categorical columns in a specific way, and activate a "categorical enabled" mode during training.
* Treats valid (ie. non-missing) and missing values using different splitting algorithms. A workflow may choose to impute missing values, or keep them as-is. The training shall yield a different XGBoost estimator for either case.

## Dataset

### Feature map

A feature map organizes feature descriptions into a table-like data structure.

Each entry consists of three fields:

1. Index. A 0-based index indicating the column index in the training dataset.
2. Name. A string indicating the column name in the training dataset. For binary indicator features, the name follows the pattern `<column name>=<value>`.
3. Type code. A string enum, either `q` (quantitative), `i` (binary indicator) or `c` (categorical). The `q` type code may be replaced with `float` (floating-point number) or `int` (integer number) type codes for extra detail.

Entries are ordered by index, and stored each entry on a line by itself, using the tab character (`\t`) as field separator.
This layout is identical to the headerless tab-separated values (TSV) data format.

Feature maps have lost relevance since XGBoost switched to a JSON-based model data format.
The booster object now incorporates the same information directly using its `feature_names` and `feature_types` attributes, enabling automated data schema checks during prediction.

Unfortunately, XGBoost still cannot help with the encoding of categorical features.
The booster object refers to category levels by their indices, but does not store the underlying mapping itself for reference.
Users must find their own ways for working around this deficiency.

The JPMML-XGBoost library relies on feature maps, where categorical features are represented as lists of binary indicator features. The position of each category level in the list indicates its index.

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
* `--target-categories`. The comma-separated list of class labels of the target field. Defaults to 0-based indices `[0, 1, ..., n_classes - 1]`.

For example, converting a binary classifier with a custom target field:

```bash
java -jar pmml-xgboost-example-executable-${version}.jar --model-input Booster.json --fmap-input Booster.fmap.tsv --pmml-output Booster.pmml --target-name "event" --target-categories "no,yes"
```
