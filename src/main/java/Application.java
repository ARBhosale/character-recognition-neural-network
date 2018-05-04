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
        Instances dataset = loadDataSet(Application.TWOCLASS_TRAINING_DATA_PATH);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        // divide dataset to train dataset 80% and test dataset 20%
        int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
        int testSize = dataset.numInstances() - trainSize;
        dataset.randomize(new Debug.Random(1));
        filter.setInputFormat(dataset);
        Instances normalizedDataset = Filter.useFilter(dataset, filter);

        Instances normalizedTrainingSet = new Instances(normalizedDataset, 0, trainSize);
        Instances normalizedValidationSet = new Instances(normalizedDataset, trainSize, testSize);

        dataset.delete();
        long endTime = System.nanoTime();
        System.out.println("Training samples: " + normalizedTrainingSet.size() + "\nValidation samples: " + normalizedValidationSet.size() + "\nTime taken: " + (endTime - startTime) / 1000000 + " milliseconds");

        TestConfigs testConfigs = new TestConfigs();
        NeuralNetworkConfig testConfig = testConfigs.getNetworkConfigs().get(0);

        NeuralNetwork network = new NeuralNetwork(testConfig, normalizedTrainingSet, normalizedValidationSet);
        network.train();
        System.out.println(network.toString());

        System.out.println("Loading testing data...");
        startTime = System.nanoTime();
        Instances testingSet = loadDataSet(Application.TWOCLASS_TESTING_DATA_PATH);
        testingSet.randomize(new Debug.Random(1));
        filter.setInputFormat(testingSet);
        Instances normalizedTestingSet = Filter.useFilter(testingSet, filter);
        normalizedTestingSet.setClassIndex(normalizedTestingSet.numAttributes() - 1);
        testingSet.delete();
        endTime = System.nanoTime();
        System.out.println("Testing samples: " + normalizedTestingSet.size() + "\nTime taken: " + (endTime - startTime) / 1000000 + " milliseconds");
        System.out.println("Predicting...");
        for (Instance instance : normalizedTestingSet) {
            network.predictClass(instance);
        }

        long countOfCorrectPredictions = network.getCountOfCorrectPredictions();
        System.out.println("Accuracy : " + (Double.parseDouble(countOfCorrectPredictions + "") / normalizedTestingSet.size()) + " (" + countOfCorrectPredictions + "/" + normalizedTestingSet.size() + ")");
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
