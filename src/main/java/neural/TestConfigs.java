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

        HiddenLayerConfig hConfig1 = new HiddenLayerConfig(150, 's');
        hConfig1.setThreshold(0.6d);

        HiddenLayerConfig hConfig2 = new HiddenLayerConfig(100, 's');
        hConfig2.setThreshold(0.6d);

        ArrayList<HiddenLayerConfig> hConfigs = new ArrayList<HiddenLayerConfig>();

        hConfigs.add(hConfig1);
//        hConfigs.add(hConfig2);

        NeuralNetworkConfig config1 = new NeuralNetworkConfig(hConfigs);
        config1.setInputLayerThreshold(0.6d);
        config1.setOutputLayerThreshold(0.6d);
        config1.setOutputLayerTransformFunction('x');
        config1.setLearningRate(0.5);
        config1.setNumberOfEpochs(100);
        this.networkConfigs.add(config1);
    }
}
