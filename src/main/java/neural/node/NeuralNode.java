package neural.node;

import neural.connection.NeuralConnection;

import java.util.ArrayList;

public class NeuralNode {
    private static Long count = 0l;
    private Long id;
    private Long layerId;
    private ArrayList<NeuralConnection> incomingConnections;
    private ArrayList<NeuralConnection> outgoingConnections;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getLayerId() {
        return layerId;
    }

    public void setLayerId(Long layerId) {
        this.layerId = layerId;
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

    public NeuralNode(Long layerId) {
        this.layerId = layerId;
        this.id = ++NeuralNode.count;
        this.incomingConnections = new ArrayList<NeuralConnection>();
        this.outgoingConnections = new ArrayList<NeuralConnection>();
    }
}
