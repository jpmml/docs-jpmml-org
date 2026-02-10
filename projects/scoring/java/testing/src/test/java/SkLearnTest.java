import org.jpmml.evaluator.testing.PMMLEquivalence;
import org.jpmml.evaluator.testing.SimpleArchiveBatchTest;
import org.junit.jupiter.api.Test;

public class SkLearnTest extends SimpleArchiveBatchTest {

	public SkLearnTest(){
		super(new PMMLEquivalence(1e-13, 1e-13));
	}

	@Test
	public void evaluateDecisionTreeIris() throws Exception {
		evaluate("DecisionTree", "Iris");
	}

	@Test
	public void evaluateLogisticRegressionIris() throws Exception {
		evaluate("LogisticRegression", "Iris");
	}
}