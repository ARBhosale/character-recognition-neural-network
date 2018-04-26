package neural.network;

import neural.connection.NeuralConnection;
import neural.layer.NeuralLayer;
import neural.node.NeuralNode;

import java.util.ArrayList;

public class NeuralNetwork {

    private NeuralNetworkConfig networkConfig;
    private NeuralLayer inputLayer;
    private ArrayList<NeuralLayer> hiddenLayers;
    private NeuralLayer outputLayer;

    public NeuralNetwork(NeuralNetworkConfig networkConfig) {
        this.networkConfig = networkConfig;
        this.initializeNeuralLayers();
        this.initializeNeuralConnections();
    }

    private void initializeNeuralLayers() {
        this.initializeInputLayer();
        this.initializeHiddenLayer();
        this.initializeOutputLayer();
    }

    private void initializeNeuralConnections() {
        if (this.hiddenLayers.size() == 0) {
            this.initializeConnections(this.inputLayer, this.outputLayer);
            return;
        }
        this.initializeConnections(this.inputLayer, this.hiddenLayers.get(0));
        for (int i = 1; i < this.hiddenLayers.size() - 1; i++) {
            this.initializeConnections(this.hiddenLayers.get(i), this.hiddenLayers.get(i + 1));
        }
        this.initializeConnections(this.hiddenLayers.get(this.hiddenLayers.size() - 1), this.outputLayer);
    }


    private void initializeInputLayer() {
        this.inputLayer = new NeuralLayer(this.networkConfig.getNumberOfInputUnits());
    }


    private void initializeHiddenLayer() {
        ArrayList<HiddenLayerConfig> hConfigs = this.networkConfig.getHiddenLayerConfigs();
        this.hiddenLayers = new ArrayList<NeuralLayer>(hConfigs.size());
        for (HiddenLayerConfig config : hConfigs) {
            NeuralLayer hiddenLayer = new NeuralLayer(config.getNumberOfHiddenUnits());
            this.hiddenLayers.add(hiddenLayer);
        }
    }

    private void initializeOutputLayer() {
        this.outputLayer = new NeuralLayer(this.networkConfig.getNumberOfOutputUnits());
    }

    //    connections from Layer A to Layer B
    private void initializeConnections(NeuralLayer layerA, NeuralLayer layerB) {
        for (NeuralNode nodeA : layerA.getNeuralNodes()) {
            for (NeuralNode nodeB : layerB.getNeuralNodes()) {
                NeuralConnection connection = new NeuralConnection(nodeA, nodeB);
                nodeA.getOutgoingConnections().add(connection);
                nodeB.getIncomingConnections().add(connection);
            }
        }
    }

}
