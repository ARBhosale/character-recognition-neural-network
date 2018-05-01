import neural.TestConfigs;
import neural.network.NeuralNetwork;
import neural.network.NeuralNetworkConfig;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Application {

    private static final String TWOCLASS_TRAINING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-trainFor2Class.arff";
    private static final String TWOCLASS_TESTING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-testFor2Class.arff";

    private static final String TRAINING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-train3.arff";
    private static final String TESTING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-test3.arff";

    public static void main(String[] args) throws Exception {
        System.out.println("Start!");
        Filter filter = new Normalize();

        System.out.println("Loading training data...");
        long startTime = System.nanoTime();
        Instances trainingSet = loadDataSet(Application.TWOCLASS_TRAINING_DATA_PATH);
        trainingSet.randomize(new Debug.Random(1));
        filter.setInputFormat(trainingSet);
        Instances normalizedTrainingSet = Filter.useFilter(trainingSet, filter);
        normalizedTrainingSet.setClassIndex(normalizedTrainingSet.numAttributes() - 1);
        trainingSet.delete();
        long endTime = System.nanoTime();
        System.out.println("Training data with " + normalizedTrainingSet.size() + " instances prepared in " + (endTime - startTime) / 1000000 + " milliseconds");


        System.out.println("Loading testing data...");
        startTime = System.nanoTime();
        Instances testingSet = loadDataSet(Application.TWOCLASS_TESTING_DATA_PATH);
        testingSet.randomize(new Debug.Random(1));
        filter.setInputFormat(testingSet);
        Instances normalizedTestingSet = Filter.useFilter(testingSet, filter);
        normalizedTestingSet.setClassIndex(normalizedTestingSet.numAttributes() - 1);
        testingSet.delete();
        endTime = System.nanoTime();
        System.out.println("Testing data with " + normalizedTestingSet.size() + " instances prepared in " + (endTime - startTime) / 1000000 + " milliseconds");


        TestConfigs testConfigs = new TestConfigs();
        NeuralNetworkConfig testConfig = testConfigs.getNetworkConfigs().get(0);

        NeuralNetwork network = new NeuralNetwork(testConfig, normalizedTrainingSet, normalizedTestingSet);
        network.train();
        System.out.println(network.toString());


        System.out.println("Predicting...");
        for (Instance instance : normalizedTestingSet) {
            network.predictClass(instance);
        }
//        network.predictClass(normalizedTestingSet.get(0));
//        network.predictClass(normalizedTestingSet.get(155));
//        network.predictClass(normalizedTestingSet.get(345));
//        network.predictClass(normalizedTestingSet.get(450));
//        network.predictClass(normalizedTestingSet.get(565));
//        network.predictClass(normalizedTestingSet.get(689));

        System.out.println("Correctly predicted : " + network.getCountOfCorrectPredictions() + " instances out of " + normalizedTestingSet.size());
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
