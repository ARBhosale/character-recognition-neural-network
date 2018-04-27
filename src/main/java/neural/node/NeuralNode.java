package neural.node;

import neural.connection.NeuralConnection;
import neural.layer.NeuralLayer;

import java.util.ArrayList;
import java.util.concurrent.Callable;

public class NeuralNode {
    private static Long count = 0l;
    private Long id;
    private NeuralLayer layer;

    private ArrayList<NeuralConnection> incomingConnections;
    private ArrayList<NeuralConnection> outgoingConnections;

    private Double activatedValue = 0D;
    private Double thresholdValue = 0D;
    private Double outputValue = 0D;
    private char transformFunction;

    public char getTransformFunction() {
        return transformFunction;
    }

    public void setTransformFunction(char transformFunction) {
        this.transformFunction = transformFunction;
    }

    public void activateNeuron() {
        if (null == this.incomingConnections) {
            return;
        }
        // for an incoming connection, this node is nodeB
        for (NeuralConnection connection : this.incomingConnections) {
            this.activatedValue += connection.getWeight() * connection.getNodeA().getOutputValue();
        }
        this.activatedValue -= this.thresholdValue;
    }

    public void transformNeuron() {
        switch (this.transformFunction) {
            case 's':
                this.outputValue = this.transformSigmoid();
                break;
            default:
                this.outputValue = this.activatedValue;
        }
    }

    public Double getThresholdValue() {
        return thresholdValue;
    }

    public void setThresholdValue(Double thresholdValue) {
        this.thresholdValue = thresholdValue;
    }

    public Double getOutputValue() {
        return outputValue;
    }

    public void setOutputValue(Double outputValue) {
        this.outputValue = outputValue;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public NeuralLayer getLayer() {
        return layer;
    }

    public void setLayer(NeuralLayer layer) {
        this.layer = layer;
    }

    public ArrayList<NeuralConnection> getIncomingConnections() {
        return incomingConnections;
    }

    public void setIncomingConnections(ArrayList<NeuralConnection> incomingConnections) {
        this.incomingConnections = incomingConnections;
    }

    public ArrayList<NeuralConnection> getOutgoingConnections() {
        return outgoingConnections;
    }

    public void setOutgoingConnections(ArrayList<NeuralConnection> outgoingConnections) {
        this.outgoingConnections = outgoingConnections;
    }

    public NeuralNode(NeuralLayer layer) {
        this.layer = layer;
        this.id = ++NeuralNode.count;
        this.incomingConnections = new ArrayList<NeuralConnection>();
        this.outgoingConnections = new ArrayList<NeuralConnection>();
        this.transformFunction = layer.getTransformFunction();
        this.thresholdValue = layer.getThreshold();
    }

    public void updateValue(Double delta) {
        this.outputValue += delta;
    }

    private Double transformSigmoid() {
        return 1 / (1 + (Math.exp(this.activatedValue)));
    }
}
