package neural.network;

import java.util.ArrayList;

public class NeuralNetworkConfig {
    private Integer numberOfInputUnits;
    private Integer numberOfOutputUnits;
    private ArrayList<HiddenLayerConfig> hiddenLayerConfigs;

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
