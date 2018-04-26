package neural.connection;

import neural.node.NeuralNode;

// connection from Node A to Node B
public class NeuralConnection {
    private static Long count = 0l;
    private Long id;
    private NeuralNode nodeA;
    private NeuralNode nodeB;
    private Double weight;

    public NeuralConnection(NeuralNode nodeA, NeuralNode nodeB) {
        this.nodeA = nodeA;
        this.nodeB = nodeB;
        this.id = ++NeuralConnection.count;
    }
}
