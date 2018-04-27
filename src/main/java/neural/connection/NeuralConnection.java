package neural.connection;

import neural.node.NeuralNode;

import java.util.Random;

// connection from Node A to Node B
public class NeuralConnection {
    private static Long count = 0l;
    private Long id;
    private NeuralNode nodeA;
    private NeuralNode nodeB;
    private Double weight = Math.random();

    public NeuralConnection(NeuralNode nodeA, NeuralNode nodeB) {
        this.nodeA = nodeA;
        this.nodeB = nodeB;
        this.id = ++NeuralConnection.count;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public NeuralNode getNodeA() {
        return nodeA;
    }

    public void setNodeA(NeuralNode nodeA) {
        this.nodeA = nodeA;
    }

    public NeuralNode getNodeB() {
        return nodeB;
    }

    public void setNodeB(NeuralNode nodeB) {
        this.nodeB = nodeB;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }
}
