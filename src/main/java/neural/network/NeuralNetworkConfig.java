package neural.network;

import java.util.ArrayList;

public class NeuralNetworkConfig {
    //    input layer
    private Integer numberOfInputUnits;
    private char inputLayerTransformFunction;
    private Double inputLayerThreshold = 0D;
    //    output layer
    private Integer numberOfOutputUnits;
    private char outputLayerTransformFunction;
    private Double outputLayerThreshold = 0D;
    //    hidden layer
    private ArrayList<HiddenLayerConfig> hiddenLayerConfigs;

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

    public NeuralNetworkConfig(Integer numberOfInputUnits, Integer numberOfOutputUnits, ArrayList<HiddenLayerConfig> hiddenLayerConfigs) {
        this.numberOfInputUnits = numberOfInputUnits;
        this.numberOfOutputUnits = numberOfOutputUnits;
        this.hiddenLayerConfigs = hiddenLayerConfigs;
    }

    public Integer getNumberOfInputUnits() {
        return numberOfInputUnits;
    }

    public void setNumberOfInputUnits(Integer numberOfInputUnits) {
        this.numberOfInputUnits = numberOfInputUnits;
    }

    public Integer getNumberOfOutputUnits() {
        return numberOfOutputUnits;
    }

    public void setNumberOfOutputUnits(Integer numberOfOutputUnits) {
        this.numberOfOutputUnits = numberOfOutputUnits;
    }

    public ArrayList<HiddenLayerConfig> getHiddenLayerConfigs() {
        return hiddenLayerConfigs;
    }

    public void setHiddenLayerConfigs(ArrayList<HiddenLayerConfig> hiddenLayerConfigs) {
        this.hiddenLayerConfigs = hiddenLayerConfigs;
    }
}
