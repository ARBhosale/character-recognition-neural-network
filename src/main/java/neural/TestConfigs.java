package neural;

import neural.network.HiddenLayerConfig;
import neural.network.NeuralNetworkConfig;

import java.util.ArrayList;

public class TestConfigs {
    private ArrayList<NeuralNetworkConfig> networkConfigs;
    private Integer numberOfInputUnits;
    private Integer numberOfOutputUnits;

    public TestConfigs() {
        this.initializeTestConfigs();
    }

    public ArrayList<NeuralNetworkConfig> getNetworkConfigs() {
        return networkConfigs;
    }

    public void setNetworkConfigs(ArrayList<NeuralNetworkConfig> networkConfigs) {
        this.networkConfigs = networkConfigs;
    }

    private void initializeTestConfigs() {
        this.networkConfigs = new ArrayList<NeuralNetworkConfig>(1);
        this.initializeTestConfig1();
    }

    private void initializeTestConfig1() {
        Integer numberOfHiddenUnits = 50;
        HiddenLayerConfig hConfig1 = new HiddenLayerConfig(numberOfHiddenUnits, 's');
        hConfig1.setThreshold(0.1D);
        ArrayList<HiddenLayerConfig> hConfigs = new ArrayList<HiddenLayerConfig>();
        hConfigs.add(hConfig1);

        NeuralNetworkConfig config1 = new NeuralNetworkConfig(hConfigs);
        config1.setInputLayerThreshold(0.1);
        config1.setOutputLayerThreshold(0.1);
        config1.setOutputLayerTransformFunction('s');
        config1.setLearningRate(0.9);
        config1.setNumberOfEpochs(100);
        this.networkConfigs.add(config1);
    }
}
