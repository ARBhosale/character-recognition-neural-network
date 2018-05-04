package neural.node;

import neural.connection.NeuralConnection;
import neural.layer.NeuralLayer;

import java.util.ArrayList;

public class NeuralNode {
    private static Long count = 0L;
    private Long id;
    private NeuralLayer layer;

    private ArrayList<NeuralConnection> incomingConnections;
    private ArrayList<NeuralConnection> outgoingConnections;

    private double activatedValue = 0d;
    private double thresholdValue = 0d;
    private double outputValue = 0d;
    private double targetValue;

    private double errorValue = 0d;

    private char transformFunction;

    public double getActivatedValue() {
        return activatedValue;
    }

    public void setActivatedValue(double activatedValue) {
        this.activatedValue = activatedValue;
    }

    public double getTargetValue() {
        return targetValue;
    }

    public void setTargetValue(double targetValue) {
        this.targetValue = targetValue;
    }

    public double getErrorValue() {
        return errorValue;
    }

    public void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

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
        this.activatedValue = 0d;
        long normalizer = 0l;
        // for an incoming connection, this node is nodeB
        for (NeuralConnection connection : this.incomingConnections) {
            if (connection.getNodeA().getOutputValue() > 0) {
                normalizer++;
            }
            this.activatedValue += connection.getWeight() * connection.getNodeA().getOutputValue();
        }
        this.activatedValue = (this.activatedValue / normalizer) - this.thresholdValue;
//        this.activatedValue -= this.thresholdValue;
//        this.activatedValue = (this.activatedValue / this.incomingConnections.size()) - this.thresholdValue;
    }

    public void transformNeuron() {
        if (null == this.incomingConnections) {
            return;
        }
        switch (this.transformFunction) {
            case 's':
                this.outputValue = this.transformSigmoid(this.activatedValue);
                break;
            case 'x':
                this.outputValue = this.transformSoftmax(this.activatedValue);
                break;
            default:
                this.outputValue = this.activatedValue;
        }
    }

    public void backPropagateError() {
        switch (this.transformFunction) {
            case 's':
                this.errorValue = this.getErrorValue(this.getDerivativeForSigmoid(this.outputValue));
                break;
            case 'x':
                this.errorValue = this.getErrorValue(this.getDerivativeForSoftmax(this.outputValue));
                break;
            default:
                this.errorValue = this.getErrorValue(1d);
        }
    }

    public double getThresholdValue() {
        return thresholdValue;
    }

    public void setThresholdValue(double thresholdValue) {
        this.thresholdValue = thresholdValue;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public void setOutputValue(double outputValue) {
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
        this.initializeNeuralNode();
    }

    public NeuralNode(NeuralLayer layer, double targetValue) {
        this.layer = layer;
        this.id = ++NeuralNode.count;
        this.targetValue = targetValue;
        this.initializeNeuralNode();
    }

    public void updateValue(double delta) {
        this.outputValue += delta;
    }

    private void initializeNeuralNode() {
        this.incomingConnections = new ArrayList<NeuralConnection>();
        this.outgoingConnections = new ArrayList<NeuralConnection>();
        this.transformFunction = layer.getTransformFunction();
        this.thresholdValue = layer.getThreshold();
    }

    private double getErrorValue(double delta) {
        double currentErrorValue = 0d;
        if (this.outgoingConnections.isEmpty()) {
            //  output node
            currentErrorValue = (this.targetValue - this.outputValue) * delta;
        } else {
            for (NeuralConnection connection : this.outgoingConnections) {
                //  this node is node A in the connection
                NeuralNode nodeConnectedTo = connection.getNodeB();
                currentErrorValue += connection.getWeight() * nodeConnectedTo.getErrorValue() * delta;
            }
        }
        return currentErrorValue;
    }

    private double transformSoftmax(double value) {
        double totalLayerInput = 0d;

        for (NeuralNode node : this.layer.getNeuralNodes()) {
            totalLayerInput += Math.exp(node.getActivatedValue());
        }

        double output = Math.exp(value) / totalLayerInput;
        return output;
    }

    private double getDerivativeForSoftmax(double value) {
        return value * (1d - value);
    }

    private double transformSigmoid(double value) {

        return 1 / (1 + (Math.exp(-1 * value)));
    }

    private double getDerivativeForSigmoid(double value) {
        return value * (1d - value);
    }
}
