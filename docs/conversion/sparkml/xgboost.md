---
title: Converting Apache Spark XGBoost pipelines to PMML
notebook: conversion/sparkml/xgboost.ipynb
---

## Dataset

### Categorical columns

XGBoost is similar to Apache Spark's decision tree models in that it supports direct categorical splits.

Apache Spark estimators automatically distinguish between categorical and continuous features by inspecting whether the backing column has a `NominalAttribute` metadata field.

In contrast, XGBoost does not rely on such automation.
Instead, the data scientist must explicitly declare feature types using the `XGBoostParams#setFeatureTypes(Array[String])` method:

```scala
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

val catCols: Array[String] = ???
val contCols: Array[String] = ???

def getCatTypeCode(df: DataFrame, colName: String): String = {
	df.schema(colName).dataType match {
		case _ => "c"
	}
}

def getContTypeCode(df: DataFrame, colName: String): String = {
	df.schema(colName).dataType match {
		case IntegerType => "int"
		case FloatType | DoubleType => "float"
		case _ => "q"
	}
}

val featureTypes: Array[String] = catCols.map(catCol => getCatTypeCode(df, catCol)) ++ contCols.map(contCol => getContTypeCode(df, contCol))
```

If feature types are not declared, then XGBoost assumes numeric continuous features.
This has a huge detrimental effect especially on `StringIndexer(Model)` output columns, because numeric splits would treat category level popularity rankings as meaningful ordinal values, generating totally arbitrary category groupings.

To reiterate - XGBoost does not perform any feature proofing on its own.
It will almost always produce some kind of model, and the data scientist must analyze it afterwards in order to confirm that all the intended categorical features were detected and treated as such.

This analysis goes beyond observing numeric statistics.

The surest option is to dump the underlying booster object in text data format and manually inspect the type of splits (ie. categorical vs numeric) generated for each feature in the dataset.
Check the first few trees.
If a feature exhibits correct split behaviour in one tree, then XGBoost's centralized feature type definitions guarantee the same correct split behaviour across all trees.

### Continuous columns

Cast columns to the most specific (numeric-) data type.
For example, distinguish integer columns from floating-point columns by casting them to `IntegerType`.

XGBoost respects feature type hints, and stores them permanently in the trained booster object.
For example, when dumping a booster in text data format (eg. JSON or TXT), then the threshold values for integer features are rounded (using a modified floor algorithm), whereas they remain fractional for floating-point features.

### Missing values

Apache Spark's SQL layer treats both `null` (ie. SQL `NULL`) and `NaN` values as first-class citizens.
However, the same cannot be said about its `spark.ml` and `spark.mllib` library layers.

Most built-in transformers and estimators respond poorly to missing values.
They typically require rows with `null` or `NaN` values to be dropped from the dataset, or replaced with valid values, eg. via imputation.

In contrast, XGBoost has offered excellent missing value support since its inception.
When working with high sparsity datasets, this capability may become a strong argument for choosing XGBoost over any alternative.

The choice of the sentinel value depends on column data types.
The `null` value can be used with any scalar data type, whereas the `NaN` value can only be used with floating-point data types (eg. `FloatType`, `DoubleType`).

Make the decision as early as possible, such as when loading the dataset, and stick with it throughout the pipeline.

Loading "N/A" cells as `null` values:

```scala
val df = spark.read
	.format("csv")
	.option("nullValue", "N/A")
	.read("Dataset.csv")
```

Loading "N/A" cells as `NaN` values:

```scala
val df = spark.read
	.format("csv")
	.option("nanValue", "N/A")
	.read("Dataset.csv")
```

### Math context

XGBoost is unique among ML algorithms because it uses `float32` both for data representation and math operations.

When running XGBoost on top of a `float64`-oriented ML framework such as Apache Spark, it appears to make predictions (eg. classification probabilities, regression labels) also on `float64` scale.
However, all this "extra precision" is non-substantial beyond fifth or sixth significant digit, because it was added there by the `float32` to `float64` widening cast. 

Truncating XGBoost regression results to their natural precision:

```scala
import org.apache.spark.sql.types.FloatType

val predictionCol = regressor.getPredictionCol

val predDf = pipelineModel.transform(df)
	.withColumn(predictionCol, col(predictionCol).cast(FloatType))
```

## Workflow

### Dense datasets

Use the `StringIndexer` transformer to ordinally encode all columns that are destined to become categorical features.
Any ordering strategy (eg. by frequency, lexicographical or explicit) works equally well.
The chosen ordering only affects the visual appearance of categorical splits in booster dump files.

Despite the "String" prefix in its class name, the `StringIndexer` transformer can handle any scalar data type column.
Even better, it can process an arbitrary number of scalar columns in one go.

The most critical stage is vector assembly, because the data scientist must keep track whether each vector element represents a categorical or continuous feature.

Use a single `StringIndexer(Model)` transformer for all categorical features, and assign its output columns to leading feature vector positions (ie. `0` through `#numOutputCols - 1`).
Creating the feature types array is now as easy as mapping these indices to the categorical type code, and all the remaining indices to maximally specific numeric type codes:

```scala
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

val catIndexer = new StringIndexer()
	.setInputCols(catCols)
	.setOutputCols(catCols.map(_ + "Indexed"))

val vecAssembler = new VectorAssembler()
	.setInputCols(catIndexer.getOutputCols ++ contCols)

val featureTypes: Array[String] = catIndexer.getOutputCols.map(_ => "c") ++ contCols.map(_ => "q")
```

The `VectorAssembler` transformer may return dense vectors for some rows, and sparse vectors for others.
The vector representation is decided by internal heuristics. There is no param for enforcing consistency (ie. all dense or all sparse).

This poses a serious problem, because XGBoost estimators accept dense vectors, but reject sparse vectors on the grounds that their content is prone to mis-interpretation.

The JPMML-SparkML library provides the `org.jpmml.sparkml.feature.VectorDensifier` transformer for unifying any vector representation to a dense vector.

Densifying feature vectors for XGBoost estimator needs:

```scala
import org.jpmml.sparkml.feature.VectorDensifier
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor

val vecDensifier = new VectorDensifier()
	.setInputCol(vecAssembler.getOutputCol)
	.setOutputCol(vecAssembler.getOutputCol + "Dense")

val regressor = new XGBoostRegressor()
	.setFeaturesCol(vecDensifier.getOutputCol)
	.setFeatureTypes(featureTypes)
```

### Sparse datasets

The overall pipeline design does not change much when moving from dense to sparse datasets.

The key difference is that every intermediate stage must be configured to accept and pass through missing values unchanged.
A missing value must be able to reach the XGBoost estimator intact. It must not trigger any error conditions, or be transformed into a non-missing value.

Direct configuration is possible with transformer classes that mix in the `HasHandleInvalid` trait.
Simply change the `handleInvalid` param value from the default `error` to `keep`.

One might wonder whether Apache Spark provides a `HasHandleMissing` trait for this scenario.
Unfortunately, the answer is negative.
Apache Spark regards the missing value space as a "subspace" of the invalid value space, rather than giving it equal standing.
All values that fail the validity check are lumped together and covered by the `HasHandleInvalid` semantics.

Invoking `StringIndexer(Model)#setHandleInvalid("keep")` does not pass missing and invalid values through unchanged.
Instead, they are mapped to a special category level (aka bucket), which is labeled `__unknown` for string features and `-999` for numeric features.

Therefore, the `StringIndexerModel#transform(Dataset)` method has a very unfortunate side effect: it transforms a sparse input dataset into a dense output dataset.

The solution is to post-process the dataset to make it functionally sparse again.

The JPMML-SparkML library provides the `org.jpmml.sparkml.feature.InvalidCategoryTransformer` transformer, which fully undoes the ordinal encoding of missing and invalid values:

* Removes the special category level from the `NominalAttribute` metadata field.
* Replaces all special category references with `NaN`.

Combining `StringIndexer` and `InvalidCategoryTransformer` to perform missing value-aware ordinal encoding:

```scala
import org.apache.spark.ml.feature.StringIndexer
import org.jpmml.sparkml.feature.InvalidCategoryTransformer

val catIndexer = new StringIndexer()
	.setInputCols(catCols)
	.setOutputCols(catCols.map(_ + "Indexed"))
	// Map null and NaN values to a special bucket
	.setHandleInvalid("keep")

// Restore NaN values by removing the special bucket
val catIndexTransformer = new InvalidCategoryTransformer()
	.setInputCols(catIndexer.getOutputCols)
	.setOutputCols(catIndexer.getOutputCols.map(_ + "Fixed"))
```

This approach disregards idiomatic invalid values (ie. "unknown" values), because they too become `NaN` values.

If there is a need to discriminate between the three value spaces (ie. valid, missing and invalid) in full detail, then refactor the pipeline to isolate and process them one after another.

One possibility is to insert a `org.jpmml.sparkml.feature.CategoricalDomain` decorator in the beginning of the pipeline, which maps all unknown category levels into a known category level.
This guarantees that subsequent `StringIndexer(Model)` transformers cannot encounter idiomatic invalid values and, consequently, that their `#setHandleInvalid("keep")` invocations affect only missing values.

```scala
import org.apache.spark.ml.feature.StringIndexer
import org.jpmml.sparkml.feature.{CategoricalDomain, InvalidValueTreatment}

// Process invalid values
val catDomain = new CategoricalDomain()
	.setInputCols(catCols)
	.setOutputCols(catCols.map(_ + "Validated"))
	.setInvalidValueTreatment(InvalidValueTreatment.AsValue.name)
	.setInvalidValueReplacement("(other)")

// Process valid and missing values
val catIndexer = new StringIndexer()
	.setInputCols(catDomain.getOutputCols)
	.setOutputCols(catDomain.getOutputCols.map(_ + "Indexed"))
	.setHandleInvalid("keep")
```

## Export to PMML

Convert Apache Spark XGBoost pipelines to PMML using the JPMML-SparkML library:

```scala
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.sparkml.model.HasPredictionModelOptions
import org.jpmml.xgboost.HasXGBoostOptions

val pmmlBuilder = new PMMLBuilder(df.schema, pipelineModel)
	// Apache Spark options
	.putOption(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, false)
	// XGBoost options
	.putOption(HasXGBoostOptions.OPTION_INPUT_FLOAT, true)
	.putOption(HasXGBoostOptions.OPTION_COMPACT, false)

println(pmmlBuilder.buildString)
```

{{ resources_section() }}
