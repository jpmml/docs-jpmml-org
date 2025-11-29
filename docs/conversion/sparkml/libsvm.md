---
title: Converting Apache Spark LibSVM pipelines to PMML
notebook: conversion/sparkml/libsvm.ipynb
---

## Overview

The [LibSVM](https://github.com/cjlin1/libsvm) project defines a text-based data format for representing numeric datasets.

Apache Spark uses the LibSVM data format to enable "split pipeline" workflows, where data pre-processing and modeling happen in separate pipelines.

The LibSVM data format specializes in storing data pre-processing results.
It outperforms the alternatives as data size and sparsity increase (eg. high-cardinality categorical features, TF-IDF features).

However, a dataset in LibSVM data format should be regarded as a short-lived artifact for internal consumption.
It needs backing by code or documentation to explain its contents. A casual observer cannot independently tell which raw features existed in the original dataset, or whether (and how) they were transformed.

This opacity carries over to trained models, where it poses a serious challenge when converting them to the PMML representation.

The PMML converter processes the core model object easily, because it is independent of the data format.
The confusion arises when wrapping this model with an appropriate data input and output interface (ie. model schema).
The fallback assumes dummy features like `<featuresCol>[<index>]`.

The PMML converter can be assisted by supplying an _ad-hoc_ pipeline model instead, which prepends a compatible data pre-processor to the model:

```scala
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.jpmml.sparkml.PipelineModelUtil

val preProcPipelineModel: PipelineModel = ???
val libsvmPipelineModel: PipelineModel = ???

val pmmlStages: Array[PipelineStage] = preProcPipelineModel.stages ++ libsvmPipelineModel.stages

// Create a pipeline model from pre-fitted pipeline stages
val pmmlPipelineModel = PipelineModelUtil.create("pmmlPipelineModel", pmmlStages)
```

## Dataset

The LibSVM reader loads and aggregates all features into a singular `VectorUDT` column in one pass.

Loading a LibSVM dataset from file:

```scala
val df = spark.read
	.format("libsvm")
	.option("numFeatures", "4")
	.option("vectorType", "dense")
	.load("Iris.libsvm")

df.printSchema()
```

Be sure to configure the expected vector size and sparsity to achieve maximal performance.

## Workflow

A LibSVM dataset has exactly two columns, `label` and `features`, which makes it compatible with any Apache Spark predictor as-is:

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

val classifier = new LogisticRegression()

val libsvmPipeline = new Pipeline()
	.setStages(Array(classifier))

val libsvmPipelineModel = libsvmPipeline.fit(df)
```

Historically, the JPMML-SparkML library has rejected pipeline models that begin with a `VectorUDT` column.

Bypass this limitation using the "split pipeline" workflow approach.

First, create a data schema that lists all fields available for reading or writing in the deployment environment.
Convey as much detail about each field as possible.

Second, create a data pre-processing pipeline that implements data flows between this new data schema and the reference LibSVM data schema.

The minimal implementation consists of a `VectorAssembler` stage that shows which scalar columns and in which order combine into the `VectorUDT` column.
The PMML converter uses this information to resolve vector indices to data schema fields, a prerequisite for generating correctly named and typed `DataField` elements.

Emulating the `label` field:

```scala
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.sql.types.{StringType, StructType}

val irisLabelSchema = new StructType()
	.add("Species", StringType, nullable = false)

val speciesIndexerModel = new StringIndexerModel("speciesIndexerModel", Array("setosa", "versicolor", "virginica"))
	.setInputCol("Species")
	.setOutputCol("label")
```

Emulating the `features` field:

```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DoubleType, StructType}

val irisFeaturesSchema = new StructType()
	.add("Sepal.Length", DoubleType, nullable = false)
	.add("Sepal.Width", DoubleType, nullable = false)
	.add("Petal.Length", DoubleType, nullable = false)
	.add("Petal.Width", DoubleType, nullable = false)

val featuresAssembler = new VectorAssembler("featuresAssembler")
	.setInputCols(Array("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))
	.setOutputCol("features")
```

The most challenging operation in Scala is creating the final pipeline model object, because the `PipelineModel(Array[PipelineStage])` constructor is private and thus only invokable using reflection.
PySpark has no such obstacle.

Final assembly:

```scala
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.sql.types.StructType
import org.jpmml.sparkml.PipelineModelUtil

// Create new data schema, completely ignoring the LibSVM data schema
val irisSchema = new StructType(irisLabelSchema.fields ++ irisFeaturesSchema.fields)

// Create new pipeline model, by prepending new pipeline stages to LibSVM pipeline stages
val irisStages: Array[Transformer] = Array(speciesIndexerModel, featuresAssembler) ++ libsvmPipelineModel.stages

val irisPipelineModel = PipelineModelUtil.create("irisPipelineModel", irisStages)
```

## Export to PMML

Convert Apache Spark pipeline models to PMML using the JPMML-SparkML library.

```scala
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.sparkml.model.HasPredictionModelOptions

val irisPmmlBuilder = new PMMLBuilder(irisSchema, irisPipelineModel)
	// Suppress redundant output fields
	.putOption(HasPredictionModelOptions.OPTION_KEEP_PREDICTIONCOL, false)

println(irisPmmlBuilder.buildString)
```

{{ resources_section() }}
