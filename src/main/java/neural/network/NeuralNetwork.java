package neural.network;

import neural.connection.NeuralConnection;
import neural.layer.NeuralLayer;
import neural.node.NeuralNode;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.event.DocumentEvent;
import java.util.ArrayList;

public class NeuralNetwork {

    private NeuralNetworkConfig networkConfig;
    private NeuralLayer inputLayer;
    private ArrayList<NeuralLayer> hiddenLayers;
    private NeuralLayer outputLayer;
    private Instances instances;

    public NeuralNetwork(NeuralNetworkConfig networkConfig, Instances instances) {
        this.networkConfig = networkConfig;
        this.instances = instances;
        this.initializeNeuralLayers();
        this.initializeNeuralConnections();
    }

    public void train() {
        System.out.println(this.toString());
        System.out.println("Starting training");
        for (int i = 0; i < this.networkConfig.getNumberOfEpochs(); i++) {
            System.out.println("Epoch: " + i);
            for (Instance instance : this.instances) {
                this.updateInputLayer(instance);
                this.forwardPropogate();
                this.backwardPropagate();
                this.updateWeights();
            }
            System.out.println("Sum of square errors: " + this.getSumSquaredErrorFromOutputLayer());
        }
    }

    private double getSumSquaredErrorFromOutputLayer() {
        double sumOfErrors = 0D;
        for (NeuralNode outputNode : this.outputLayer.getNeuralNodes()) {
            sumOfErrors += Math.pow((outputNode.getTargetValue() - outputNode.getOutputValue()), 2);
        }
        return sumOfErrors;
    }

    private void forwardPropogate() {
        this.forwardPropogateHiddenLayers();
        this.forwardPropogateFromNeuralLayer(this.outputLayer);
    }

    private void backwardPropagate() {
        this.backPropagateFromNeuralLayer(this.outputLayer);
        this.backPropagateHiddenLayers();
    }

    private void updateWeights() {
        this.updateWeightsForLayer(this.inputLayer);
        this.updateWeightsHiddenLayers();
    }

    private void updateWeightsHiddenLayers() {
        for (NeuralLayer hiddenLayer : this.hiddenLayers) {
            this.updateWeightsForLayer(hiddenLayer);
        }
    }

    private void updateWeightsForLayer(NeuralLayer layer) {
        for (NeuralNode node : layer.getNeuralNodes()) {
            for (NeuralConnection connection : node.getOutgoingConnections()) {
                connection.updateWeight();
            }
        }
    }

    private void backPropagateHiddenLayers() {
        for (NeuralLayer hiddenLayer : this.hiddenLayers) {
            this.backPropagateFromNeuralLayer(hiddenLayer);
        }
    }

    private void backPropagateFromNeuralLayer(NeuralLayer layer) {
        for (NeuralNode node : layer.getNeuralNodes()) {
            node.backPropagateError();
        }
    }

    private void forwardPropogateHiddenLayers() {
        for (NeuralLayer hiddenLayer : this.hiddenLayers) {
            this.forwardPropogateFromNeuralLayer(hiddenLayer);
        }
    }

    private void forwardPropogateFromNeuralLayer(NeuralLayer layer) {
        for (NeuralNode node : layer.getNeuralNodes()) {
            node.activateNeuron();
            node.transformNeuron();
        }
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
        this.inputLayer = new NeuralLayer(this.instances.numAttributes() - 1,
                this.networkConfig.getInputLayerTransformFunction(),
                this.networkConfig.getInputLayerThreshold(), this.networkConfig.getLearningRate());

    }

    private void updateInputLayer(Instance instance) {
        ArrayList<NeuralNode> inputNodes = this.inputLayer.getNeuralNodes();
        for (int i = 0; i < inputNodes.size(); i++) {
            if (instance.value(i) > 1) {
                int a = 3;
            }
            inputNodes.get(i).setOutputValue(instance.value(i));
        }
    }


    private void initializeHiddenLayer() {
        ArrayList<HiddenLayerConfig> hConfigs = this.networkConfig.getHiddenLayerConfigs();
        this.hiddenLayers = new ArrayList<NeuralLayer>(hConfigs.size());
        for (HiddenLayerConfig config : hConfigs) {
            NeuralLayer hiddenLayer = new NeuralLayer(config.getNumberOfHiddenUnits(),
                    config.getTransformFunction(), config.getThreshold(), this.networkConfig.getLearningRate());
            this.hiddenLayers.add(hiddenLayer);
        }
    }

    private void initializeOutputLayer() {
        this.outputLayer = new NeuralLayer(this.instances.numClasses(),
                this.networkConfig.getOutputLayerTransformFunction(),
                this.networkConfig.getOutputLayerThreshold(),
                this.networkConfig.getLearningRate());

        for (int i = 0; i < this.instances.numClasses(); i++) {
            double transformedTargetValue = this.transformSigmoid(Double.valueOf(i));
            this.outputLayer.getNeuralNodes().get(i).setTargetValue(transformedTargetValue);
        }
//        this.outputLayer = new NeuralLayer(2,
//                this.networkConfig.getOutputLayerTransformFunction(),
//                this.networkConfig.getOutputLayerThreshold(), this.networkConfig.getLearningRate());
//        for (int i = 0; i < 2; i++) {
//            this.outputLayer.getNeuralNodes().get(i).setTargetValue(Double.valueOf(i));
//        }
    }

    private double transformSigmoid(double value) {

        return 1 / (1 + (Math.exp(-1 * value)));
    }

    //    connections from Layer A to Layer B
    private void initializeConnections(NeuralLayer layerA, NeuralLayer layerB) {
        for (NeuralNode nodeA : layerA.getNeuralNodes()) {
            for (NeuralNode nodeB : layerB.getNeuralNodes()) {
                NeuralConnection connection = new NeuralConnection(nodeA, nodeB);
                connection.setLearningRate(this.networkConfig.getLearningRate());
                nodeA.getOutgoingConnections().add(connection);
                nodeB.getIncomingConnections().add(connection);
            }
        }
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" +
                "networkConfig=" + networkConfig.toString() +
                '}';
    }
}
