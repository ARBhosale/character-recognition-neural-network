import neural.TestConfigs;
import neural.network.NeuralNetwork;
import neural.network.NeuralNetworkConfig;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.logging.Level;
import java.util.logging.Logger;

public class Application {

    private static final String TWOCLASS_TRAINING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-trainFor2Class.arff";
    private static final String TWOCLASS_TESTING_DATA_PATH = "/Users/aniketbhosale/Documents/sem4/cs 782/Project/my nn/mlcs782handwrittencharacterrecognition/datasets/emnist-balanced-testFor2Class.arff";

    public static void main(String[] args) {
        System.out.println("Start!");

//        Instances trainingSet = loadDataSet(Application.TWOCLASS_TRAINING_DATA_PATH);
//        Instances testingset = loadDataSet(Application.TWOCLASS_TESTING_DATA_PATH);
//
//        System.out.println("Data loaded");
        TestConfigs testConfigs = new TestConfigs(3,2);
        NeuralNetworkConfig testConfig = testConfigs.getNetworkConfigs().get(0);
        NeuralNetwork network = new NeuralNetwork(testConfig);

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
