package neural.network;

import neural.connection.NeuralConnection;
import neural.layer.NeuralLayer;
import neural.node.NeuralNode;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;

public class NeuralNetwork {

    private NeuralNetworkConfig networkConfig;
    private NeuralLayer inputLayer;
    private ArrayList<NeuralLayer> hiddenLayers;
    private NeuralLayer outputLayer;
    private Instances trainingInstances;
    private Instances testingInstances;
    private long countOfCorrectPredictions = 0;

    public NeuralNetwork(NeuralNetworkConfig networkConfig, Instances trainingInstances, Instances testingInstances) {
        this.networkConfig = networkConfig;
        this.trainingInstances = trainingInstances;
        this.testingInstances = testingInstances;
        this.initializeNeuralLayers();
        this.initializeNeuralConnections();
    }

    public void train() {
        System.out.println(this.toString());
        System.out.println("Starting training");

        double earlierMisClassifications = Double.POSITIVE_INFINITY;
        double currentMisClassifications = Double.POSITIVE_INFINITY;

        double minSumOfSquaresError = Double.POSITIVE_INFINITY;

        long startTime = System.nanoTime();
        for (int i = 0; i < this.networkConfig.getNumberOfEpochs(); i++) {
            System.out.println("Epoch: " + i);
            for (Instance instance : this.trainingInstances) {
                this.updateInputLayer(instance);
                this.updateTargetValueInOutputLayer(instance);
                this.forwardPropogate();
                this.backwardPropagate();
                this.updateWeights(i);
//                double sumOfSquaresError = this.getSumOfSquaresError();
//                if (sumOfSquaresError <= minSumOfSquaresError) {
//                    minSumOfSquaresError = sumOfSquaresError;
//                    this.updateWeights(i);
//                }
            }
            currentMisClassifications = this.getNumberOfMisClassifications();
            System.out.println("MisClassifications: " + currentMisClassifications);
            if (currentMisClassifications <= earlierMisClassifications) {
                earlierMisClassifications = currentMisClassifications;
            } else {
                System.out.println("MisClassifications started increasing from " + earlierMisClassifications + " to " + currentMisClassifications);
//                this.revertWeights();
//                break;
            }
        }
        long endTime = System.nanoTime();
        System.out.println("Training completed in " + (endTime - startTime) / 1000000 + " milliseconds");
    }

    public void predictClass(Instance instance) {
        this.updateInputLayer(instance);
        this.forwardPropogate();
        double expectedClass = instance.value(instance.numAttributes() - 1);
        double predictedClass = this.getMaxOutputFromOutputLayer();
        if (expectedClass == predictedClass) {
            this.countOfCorrectPredictions++;
        }
//        System.out.println("Expected class: " + expectedClass + " Predicted class: " + predictedClass);
    }

    public long getCountOfCorrectPredictions() {
        return countOfCorrectPredictions;
    }

    public void setCountOfCorrectPredictions(long countOfCorrectPredictions) {
        this.countOfCorrectPredictions = countOfCorrectPredictions;
    }

    private long getNumberOfMisClassifications() {
        long misClassifications = 0l;
        for (Instance instance : this.testingInstances) {
            this.updateInputLayer(instance);
            this.forwardPropogate();
            double expectedClass = instance.value(instance.numAttributes() - 1);
            double predictedClass = this.getMaxOutputFromOutputLayer();
            if (expectedClass != predictedClass) {
                misClassifications++;
            }
        }
        return misClassifications;
    }

    private void updateTargetValueInOutputLayer(Instance instance) {
//        double targetValue = Double.parseDouble(instance.classValue().toString());
        for (int i = 0; i < this.outputLayer.getNumberOfNodes(); i++) {
            NeuralNode outputNode = this.outputLayer.getNeuralNodes().get(i);
            if (instance.classValue() == i) {
                outputNode.setTargetValue(1d);
            } else {
                outputNode.setTargetValue(0d);
            }
        }
    }

    private double getMaxOutputFromOutputLayer() {
        double max = Double.NEGATIVE_INFINITY;
        int outputClass = 0;
        for (int i = 0; i < this.outputLayer.getNumberOfNodes(); i++) {
            if (this.outputLayer.getNeuralNodes().get(i).getOutputValue() > max) {
                max = this.outputLayer.getNeuralNodes().get(i).getOutputValue();
                outputClass = i;
            }
        }
        return outputClass;
    }

    private double getSumOfSquaresError() {
        double sumOfErrors = 0d;
        for (NeuralNode outputNode : this.outputLayer.getNeuralNodes()) {
            sumOfErrors += Math.pow(outputNode.getErrorValue(), 2);
        }
        return sumOfErrors;
    }

    private double getSumSquaredErrorFromOutputLayer() {
        double sumOfErrors = 0d;
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

    private void revertWeights() {
        this.revertWeightsForLayer(this.inputLayer);
        this.revertWeightsHiddenLayers();
    }

    private void updateWeights(int epochNo) {
        this.updateWeightsForLayer(this.inputLayer, epochNo);
        this.updateWeightsHiddenLayers(epochNo);
    }

    private void updateWeightsHiddenLayers(int epochNo) {
        for (NeuralLayer hiddenLayer : this.hiddenLayers) {
            this.updateWeightsForLayer(hiddenLayer, epochNo);
        }
    }

    private void revertWeightsHiddenLayers() {
        for (NeuralLayer hiddenLayer : this.hiddenLayers) {
            this.revertWeightsForLayer(hiddenLayer);
        }
    }

    private void updateWeightsForLayer(NeuralLayer layer, int epochNo) {
        for (NeuralNode node : layer.getNeuralNodes()) {
            for (NeuralConnection connection : node.getOutgoingConnections()) {
                connection.updateWeight(epochNo);
            }
        }
    }

    private void revertWeightsForLayer(NeuralLayer layer) {
        for (NeuralNode node : layer.getNeuralNodes()) {
            for (NeuralConnection connection : node.getOutgoingConnections()) {
                connection.revertWeight();
            }
        }
    }

    private void backPropagateHiddenLayers() {
        for (int i = this.hiddenLayers.size() - 1; i >= 0; i--) {
            this.backPropagateFromNeuralLayer(this.hiddenLayers.get(i));
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
        for (int i = 0; i < this.hiddenLayers.size() - 1; i++) {
            this.initializeConnections(this.hiddenLayers.get(i), this.hiddenLayers.get(i + 1));
        }
        this.initializeConnections(this.hiddenLayers.get(this.hiddenLayers.size() - 1), this.outputLayer);
    }


    private void initializeInputLayer() {
        this.inputLayer = new NeuralLayer(this.trainingInstances.numAttributes() - 1,
                this.networkConfig.getInputLayerTransformFunction(),
                this.networkConfig.getInputLayerThreshold(), this.networkConfig.getLearningRate());

    }

    private void updateInputLayer(Instance instance) {
        ArrayList<NeuralNode> inputNodes = this.inputLayer.getNeuralNodes();
        for (int i = 0; i < inputNodes.size(); i++) {
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
        this.outputLayer = new NeuralLayer(this.trainingInstances.numClasses(),
                this.networkConfig.getOutputLayerTransformFunction(),
                this.networkConfig.getOutputLayerThreshold(),
                this.networkConfig.getLearningRate());
//        this.setNormalizedTargetValuesForOutputLayer();

//        for (int i = 0; i < this.trainingInstances.numClasses(); i++) {
//            double transformedTargetValue = this.transformSigmoid(Double.valueOf(i));
//            this.outputLayer.getNeuralNodes().get(i).setTargetValue(Double.valueOf(i));
//        }
//        this.outputLayer = new NeuralLayer(2,
//                this.networkConfig.getOutputLayerTransformFunction(),
//                this.networkConfig.getOutputLayerThreshold(), this.networkConfig.getLearningRate());
//        for (int i = 0; i < 2; i++) {
//            this.outputLayer.getNeuralNodes().get(i).setTargetValue(Double.valueOf(i));
//        }
    }

    private void setNormalizedTargetValuesForOutputLayer() {
        double[] targetValues = new double[this.trainingInstances.numClasses()];
        double sum = 0d;

        Enumeration<Object> classes = this.trainingInstances.classAttribute().enumerateValues();
        for (int i = 0; classes.hasMoreElements(); i++) {
            targetValues[i] = Double.parseDouble(classes.nextElement().toString());
            sum += targetValues[i];
        }

        for (int i = 0; i < this.trainingInstances.numClasses(); i++) {
//            double normalizedTargetValue = (targetValues[i] / sum);
            double normalizedTargetValue = (targetValues[i] / this.trainingInstances.numClasses());
            this.outputLayer.getNeuralNodes().get(i).setTargetValue(normalizedTargetValue);
        }

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
        return networkConfig.toString();
    }
}
