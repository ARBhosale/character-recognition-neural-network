package neural.layer;

import neural.network.NeuralNetworkConfig;
import neural.node.NeuralNode;

import java.util.ArrayList;

public class NeuralLayer {
    private static Long count = 0l;
    private Long id;
    private Integer numberOfNodes;
    private char transformFunction;
    private ArrayList<NeuralNode> neuralNodes;
    private double threshold = 0D;
    private double learningRate = 0D;

    public NeuralLayer(Integer numberOfNodes, char transformFunction, double threshold, double learningRate) {
        this.id = ++NeuralLayer.count;
        this.numberOfNodes = numberOfNodes;
        this.transformFunction = transformFunction;
        this.threshold = threshold;
        this.learningRate = learningRate;
        this.initializeNeuralNodes();
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Integer getNumberOfNodes() {
        return numberOfNodes;
    }

    public void setNumberOfNodes(Integer numberOfNodes) {
        this.numberOfNodes = numberOfNodes;
    }

    public ArrayList<NeuralNode> getNeuralNodes() {
        return neuralNodes;
    }

    public void setNeuralNodes(ArrayList<NeuralNode> neuralNodes) {
        this.neuralNodes = neuralNodes;
    }

    private void initializeNeuralNodes() {
        if (this.numberOfNodes <=0) {
            return;
        }
        this.neuralNodes = new ArrayList<NeuralNode>(this.numberOfNodes);
        for (int i = 0; i < this.numberOfNodes; i++) {
            NeuralNode node = new NeuralNode(this);
            node.setThresholdValue(this.threshold);
            neuralNodes.add(node);
        }
    }

    public char getTransformFunction() {
        return transformFunction;
    }

    public void setTransformFunction(char transformFunction) {
        this.transformFunction = transformFunction;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}
