package neural.network;

import java.util.ArrayList;

public class NeuralNetworkConfig {
    //    input layer
    private char inputLayerTransformFunction;
    private double inputLayerThreshold = 0D;
    //    output layer
    private char outputLayerTransformFunction;
    private double outputLayerThreshold = 0D;
    //    hidden layer
    private ArrayList<HiddenLayerConfig> hiddenLayerConfigs;

    private double learningRate = 0.3D;

    private Integer numberOfEpochs = 2;

    public Integer getNumberOfEpochs() {
        return numberOfEpochs;
    }

    public void setNumberOfEpochs(Integer numberOfEpochs) {
        this.numberOfEpochs = numberOfEpochs;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }


    public double getInputLayerThreshold() {
        return inputLayerThreshold;
    }

    public void setInputLayerThreshold(double inputLayerThreshold) {
        this.inputLayerThreshold = inputLayerThreshold;
    }

    public double getOutputLayerThreshold() {
        return outputLayerThreshold;
    }

    public void setOutputLayerThreshold(double outputLayerThreshold) {
        this.outputLayerThreshold = outputLayerThreshold;
    }


    public char getInputLayerTransformFunction() {
        return inputLayerTransformFunction;
    }

    public void setInputLayerTransformFunction(char inputLayerTransformFunction) {
        this.inputLayerTransformFunction = inputLayerTransformFunction;
    }

    public char getOutputLayerTransformFunction() {
        return outputLayerTransformFunction;
    }

    public void setOutputLayerTransformFunction(char outputLayerTransformFunction) {
        this.outputLayerTransformFunction = outputLayerTransformFunction;
    }

    public NeuralNetworkConfig(ArrayList<HiddenLayerConfig> hiddenLayerConfigs) {
        this.hiddenLayerConfigs = hiddenLayerConfigs;
    }

    public ArrayList<HiddenLayerConfig> getHiddenLayerConfigs() {
        return hiddenLayerConfigs;
    }

    public void setHiddenLayerConfigs(ArrayList<HiddenLayerConfig> hiddenLayerConfigs) {
        this.hiddenLayerConfigs = hiddenLayerConfigs;
    }

    private String getHiddenLayersString() {
        int layerNumber = 0;
        String result = "";
        for (HiddenLayerConfig config : this.hiddenLayerConfigs) {
            result += "Hidden layer " + (++layerNumber) + "\n" + config.toString() + "\n";
        }
        return result;
    }

    @Override
    public String toString() {
        return  "InputLayerTransformFunction=" + inputLayerTransformFunction +
                "\nInputLayerThreshold=" + inputLayerThreshold +
                "\nOutputLayerTransformFunction=" + outputLayerTransformFunction +
                "\nOutputLayerThreshold=" + outputLayerThreshold +
                "\nLearningRate=" + learningRate +
                "\nIterations=" + numberOfEpochs +
                "\nHidden layers config:\n" + this.getHiddenLayersString() +
                '}';
    }
}
