import neural.TestConfigs;
import neural.network.NeuralNetwork;
import neural.network.NeuralNetworkConfig;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Application {

    private static final String TWOCLASS_TRAINING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-trainFor2Class.arff";
    private static final String TWOCLASS_TESTING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-testFor2Class.arff";

    public static void main(String[] args) throws Exception {
        System.out.println("Start!");
        Filter filter = new Normalize();

        System.out.println("Loading training data...");
        Instances trainingSet = loadDataSet(Application.TWOCLASS_TRAINING_DATA_PATH);
        trainingSet.randomize(new Debug.Random(1));
        filter.setInputFormat(trainingSet);
        Instances normalizedTrainingSet = Filter.useFilter(trainingSet, filter);
        normalizedTrainingSet.setClassIndex(normalizedTrainingSet.numAttributes() - 1);
        System.out.println("Training data loaded");

        TestConfigs testConfigs = new TestConfigs();
        NeuralNetworkConfig testConfig = testConfigs.getNetworkConfigs().get(0);

        NeuralNetwork network = new NeuralNetwork(testConfig, normalizedTrainingSet);
        network.train();
        System.out.println(network.toString());


        System.out.println("Loading testing data...");
        Instances testingSet = loadDataSet(Application.TWOCLASS_TESTING_DATA_PATH);
        testingSet.randomize(new Debug.Random(1));
        filter.setInputFormat(testingSet);
        Instances normalizedTestingSet = Filter.useFilter(testingSet, filter);
        normalizedTestingSet.setClassIndex(normalizedTestingSet.numAttributes() - 1);
        System.out.println("Testing data loaded");

        network.predictClass(normalizedTestingSet.get(0));


    }

    private static Instances loadDataSet(String path) {
        Instances dataset = null;
        try {
            dataset = ConverterUtils.DataSource.read(path);
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getAnonymousLogger().log(Level.SEVERE, null, ex);
        }

        return dataset;
    }
}
