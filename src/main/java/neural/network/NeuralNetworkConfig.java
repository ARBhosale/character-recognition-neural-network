package neural.network;

import java.util.ArrayList;

public class NeuralNetworkConfig {
    //    input layer
    private char inputLayerTransformFunction;
    private Double inputLayerThreshold = 0D;
    //    output layer
    private char outputLayerTransformFunction;
    private Double outputLayerThreshold = 0D;
    //    hidden layer
    private ArrayList<HiddenLayerConfig> hiddenLayerConfigs;

    private Double learningRate = 0.3D;

    public Double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
    }


    public Double getInputLayerThreshold() {
        return inputLayerThreshold;
    }

    public void setInputLayerThreshold(Double inputLayerThreshold) {
        this.inputLayerThreshold = inputLayerThreshold;
    }

    public Double getOutputLayerThreshold() {
        return outputLayerThreshold;
    }

    public void setOutputLayerThreshold(Double outputLayerThreshold) {
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
}
