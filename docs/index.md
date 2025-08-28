The Java PMML API software project helps you to integrate free and open-source (FOSS) machine learning (ML) models into the Java/JVM platform.

The workflow is based around the [Predictive Model Markup Language (PMML)](https://dmg.org/pmml/v4-4-1/GeneralStructure.html) industry standard.

**JPMML conversion libraries** translate fitted ML models and pipelines from their native representation to the PMML representation.
The conversion captures the entire prediction logic -- from data pre-processing to decision post-processing -- thereby breaking any and all dependencies to the original ML platform.

PMML benefis:
* Stable in time. Any PMML model ever created is usable today.
* Minimal technical requirements. With PMML being an XML dialect, PMML models can be opened and maniplated with regular XML tools. No vendor lock-in.
* Human approachable. A PMML model can be printed on paper, and analyzed and scored manually if need be. 

**JPMML evaluation libraries** provide model introspection and scoring capabilities.
The compact and robust design of the PMML execution engine enables seamless embedding across the full Java application stack, plus Java-adjacent application stacks such as Python backends and JavaScript frontends.

All workflow components have been rigorously tested individually and in combination with one another.
In most cases, PMML models can reproduce original ML model predictions with an absolute or relative error of 1e-13 or better.
