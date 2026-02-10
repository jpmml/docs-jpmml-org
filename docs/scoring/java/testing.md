---
title: Integration testing Java PMML applications
project: scoring/java/testing
---

## Overview

JPMML software allows the deployment of non-Java ML framework artifacts on the Java/JVM platform.
Naturally, for business-critical applications, there needs to be ample proof that the quality of predictions did not deteriorate during the transition.

Such guarantees can be given and maintained long-term via integration testing.

The original ML artifact makes predictions on a reference dataset, and these results are stored as "correct answers".
This artifact is then converted to the PMML representation.
The PMML artifact also makes predictions on the same reference dataset, and the results are compared against the "correct answers" on a per-sample basis (ie. not on a population statistics basis).
The test passes if there are no differences beyond the stated acceptance criteria.

JPMML-Evaluator includes a testing harness, which implements this idea by asserting that a CSV input file scored with a PMML model file yields the expected result CSV file:

```
<Input CSV> + <PMML> == <Result CSV>
```

## Workflow

### Apache Maven configuration

Create a new directory, and initialize a tests-only [Apache Maven](https://maven.apache.org) project into it:

```bash
touch pom.xml
mkdir -p src/test/java
mkdir -p src/test/resources/csv
mkdir -p src/test/resources/pmml
```

Correct directory layout:

```
./
├── src/
│   └── test/
│       ├── java/
│       └── resources/
│           ├── csv/
│           └── pmml/
└── pom.xml
```

Configure the Project Object Model (POM) file to handle the `test` lifecycle phase.
This typically involves declaring a JUnit dependency, plus declaring `maven-compiler-plugin` and `maven-surefire-plugin` plugins.

To get started quickly, inherit everything from the `org.jpmml:jpmml-parent` parent POM:

```xml
<?xml version="1.0" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
	<modelVersion>4.0.0</modelVersion>

	<parent>
		<groupId>org.jpmml</groupId>
		<artifactId>jpmml-parent</artifactId>
		<version>${jpmml-parent.version}</version>
	</parent>

	<groupId>com.mycompany</groupId>
	<artifactId>myapp-integration-testing</artifactId>
	<version>1.0-SNAPSHOT</version>
</project>
```

The only "real" dependencies to declare are `org.jpmml:pmml-evaluator-metro` and `org.jpmml:pmml-evaluator-testing`.
Pin their version(s) to match the JPMML-Evaluator version that serves the actual business case.

```xml
<properties>
	<jpmml-evaluator.version>1.7.7</jpmml-evaluator.version>
</properties>

<dependencies>
	<dependency>
		<groupId>org.jpmml</groupId>
		<artifactId>pmml-evaluator-metro</artifactId>
		<version>${jpmml-evaluator.version}</version>
	</dependency>
	<dependency>
		<groupId>org.jpmml</groupId>
		<artifactId>pmml-evaluator-testing</artifactId>
		<version>${jpmml-evaluator.version}</version>
	</dependency>
</dependencies>
```

Consider extracting dependency versions into POM properties to make them overrideable using `-D<property>=value` command-line options.
For example, this pattern allows running the test suite with different JPMML-Evaluator versions without touching any code or configuration:

```bash
mvn -Djpmml-evaluator.version=1.7.7 test
mvn -Djpmml-evaluator.version=1.7.0 test
mvn -Djpmml-evaluator.version=1.6.11 test
```

### Resource files

Generate CSV and PMML files into `src/test/resources` subdirectories:

```
./
├── src/
│   └── test/
│       ├── java/
│       └── resources/
│           ├── csv/
│           │   ├── Iris.csv
│           │   ├── DecisionTreeIris.csv
│           │   └── LogisticRegressionIris.csv
│           └── pmml/
│               ├── DecisionTreeIris.pmml
│               └── LogisticRegressionIris.pmml
└── pom.xml
```

Each test case is identified by a two-part name `${algorithm}${dataset}`.
The `${algorithm}` part reflects the most characteristic algorithm of the ML artifact (eg. decision tree, logistic regression), while `${dataset}` reflects the dataset.

#### Input CSV

Store the input dataset in the `csv/${dataset}.csv` CSV file.
All JPMML testing harnesses expect comma (`,`) as the field separator, and treat `N/A` (Pandas/Python style) and `NA` (R style) as placeholders for missing values.

The input dataset is meant to be shared between related test cases.
For maximum clarity, it should contain only active feature columns, and never the target column.

Generating the "Iris" input CSV file:

```python
from pandas import DataFrame, Series
from sklearn.datasets import load_iris

iris = load_iris()

X = DataFrame(iris.data, columns = iris.feature_names)
y = Series(iris.target, name = "Species").map(lambda x: iris.target_names[x])

def store_csv(df, name):
	df.to_csv("csv/" + name + ".csv", header = True, index = False, sep = ",", na_rep = "N/A")

store_csv(X, "Iris")
```

#### PMML

Train the ML artifact.
Convert and store its PMML representation in the `pmml/${algorithm}${dataset}.pmml` PMML file.
It should go without saying that the test case should mimic the actual business case as closely as possible.

Generating the "DecisionTreeIris" PMML file:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml

def store_pmml(obj, name):
	sklearn2pmml(obj, "pmml/" + name + ".pmml")

classifier = DecisionTreeClassifier(random_state = 42)
classifier.fit(X, y)

store_pmml(classifier, "DecisionTreeIris")
```

#### Result CSV

Use the trained ML artifact to make predictions on the input dataset.
Store the results dataset in the `csv/${algorithm}${dataset}.csv` CSV file.

The test case should cover all predictive functionality.
For example, for probabilistic classifiers, it should check both class labels and their estimated probabilities.

The bigger and "more numeric" the results dataset is, the more effective it is at detecting and diagnosing potential error conditions.
For example, checking a class label alone would only catch a fraction of mis-predicted samples (ie. those that cross the decision boundary). If the dataset is small and the logic error behind mis-predictions is subtle enough, then there might not be any visible signal at all.

Generating the "DecisionTreeIris" CSV file:

```python
yt = DataFrame()
yt[y.name] = classifier.predict(X)
yt[["probability({})".format(cls) for cls in classifier.classes_]] = classifier.predict_proba(X)

store_csv(yt, "DecisionTreeIris")
```

### Java test class

Create a Java source file into the `src/test/java` subdirectory:

```
./
├── src/
│   └── test/
│       ├── java/
│       │   └── SkLearnTest.java
│       └── resources/
└── pom.xml
```

The class name can be anything, as long as it ends with `Test` or `Tests`.
The class must extend the `org.jpmml.evaluator.testing.SimpleArchiveBatchTest` abstract base class, and declare a default constructor that invokes `SimpleArchiveBatchTest(com.google.common.base.Equivalence equivalence)` with the desired equivalence strategy object:

```java
import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.evaluator.testing.SimpleArchiveBatchTest;

public class SkLearnTest extends SimpleArchiveBatchTest {

	public SkLearnTest(){
		super(new PMMLEquivalence(1e-13, 1e-13));
	}
}
```

Equivalence strategies differ by the handling of continuous floating-point values (ie. `continuous+float` and `continuous+double`).
All the other types of values are simply checked for equality.

JPMML-Evaluator provides two equivalence strategies:

* `org.jpmml.evaluator.testing.PMMLEquivalence(double precision, double zeroThreshold)`. Compares values against the zero threshold parameter to determine zeroness. If both values are non-zero, calculates the difference between them, and compares it against the precision parameter.
This procedure is described in full detail in the [Model verification](https://dmg.org/pmml/v4-4-1/ModelVerification.html) chapter of the PMML specification.
* `org.jpmml.evaluator.testing.RealNumberEquivalence(int tolerance)`. Calculates the difference between two values in Unit of Least Precision (ULP) terms, and compares it against the tolerance parameter.

Most test cases should work fine with `PMMLEquivalence` with both parameters set in the `1e-12` to `1e-14` range.
This configuration is guaranteed to catch all functionally significant differences, while still permitting some noise that stems from low-level floating-point operations and functions.

The test class should declare a separate method for each test case:

```java
import org.junit.jupiter.api.Test;

public class SkLearnTest extends SimpleArchiveBatchTest {

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate("DecisionTree", "Iris");
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate("LogisticRegression", "Iris");
	}
}
```

Test methods must be marked with the `org.junit.jupiter.api.Test` annotation for automated discovery.

The method name can be anything.
However, it is advisable to use meaningful and consistent names that relate directly to test cases (ie. containing `${algorithm}${dataset}` in some form), because Apache Maven prints failing method names to the console.

The method body invokes the most appropriate `ArchiveBatchTest#evaluate(String algorithm, String dataset)` method.
There are overloaded variants available, which enable tightening or loosening class-level configuration on a per-method basis, such as excluding some result fields from checking, or using a different equivalence strategy.

For example, with probabilistic classifiers, there are two or more estimated probability fields whose values sum to `1.0` by definition.
It may be beneficial to exclude the column with smallest magnitude values (because relative errors become less stable as the operands approach zero), in order to tighten the acceptance criteria for the remaining column(s).

```java
public class SkLearnTest extends SimpleArchiveBatchTest {

	public SkLearnTest(){
		super(new PMMLEquivalence(1e-13, 1e-13));
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate("LogisticRegression", "Iris", excludeColumns("probability(virginica)"), new PMMLEquivalence(1e-14, 1e-14));
	}
}
```

Test methods are able to locate and load the backing CSV and PMML resource files based on the `algorithm` and `dataset` parameter values. There is no extra wiring needed.

### Execution

Navigate to the root of the project directory (where the POM file is) and execute the `test` lifecycle phase:

```bash
mvn clean test
```

Apache Maven will collect, compile and run all Java test classes, and print a short summary to the console when done.
Additional build side-artifacts are available in the `target` directory.
Specifically, the `maven-surefire-plugin` plugin (the default handler for the `test` lifecycle phase) generates detailed TXT and XML reports into the `target/surefire-reports` directory.

During the first run, make sure that all the intended test fixtures are discovered.

This example project has one test class (`SkLearnTest`) with two test methods (`evaluateDecisionTreeIris` and `evaluateLogisticRegressionIris`).
The print-out confirms that they were all executed successfully:

```
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running SkLearnTest
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 1.556 s -- in SkLearnTest
[INFO] 
[INFO] Results:
[INFO] 
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0
[INFO] 
```

The same testing harness can be re-purposed from integration testing to basic performance testing.
Simply increase the complexity of algorithms, and increase the size of datasets so that JPMML-Evaluator "working time" outweighs Apache Maven and JUnit "working times" by a sufficient margin.
The current elapsed time of 1.556 seconds is almost all framework overhead.

For demonstration purposes, consider breaking the build by over-tightening the equivalence strategy from `PMMLEquivalence(1e-13, 1e-13)` to `PMMLEquivalence(1e-15, 1e-15)`.

The print-out becomes:

```
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running SkLearnTest
[ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 1.606 s <<< FAILURE! -- in SkLearnTest
[INFO] 
[INFO] Results:
[INFO] 
[ERROR] Failures: 
[ERROR]   SkLearnTest.evaluateLogisticRegressionIris:18->ArchiveBatchTest.evaluate:41->ArchiveBatchTest.evaluate:63->BatchTest.evaluate:37->BatchTest.checkConflicts:48 Found 64 conflict(s)
[INFO] 
[ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0
[INFO] 
```

The `evaluateDecisionTreeIris` test method keeps passing, but the `evaluateLogisticRegressionIris` test method fails after having detected 64 conflicts.

A test class can install a custom conflict handler by overriding the `BatchTest#checkConflicts(List<Conflict> conflicts)` method.
The default handler prints conflicts to the system error stream, and then fails by throwing an `org.opentest4j.AssertionFailedError`.

Sanitize the Apache Maven build output by redirecting all conflicts to a separate file:

```bash
mvn clean test 2> conflicts.txt
```

{{ resources_section() }}