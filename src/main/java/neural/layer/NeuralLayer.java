package neural.layer;

import neural.node.NeuralNode;

import java.util.ArrayList;

public class NeuralLayer {
    private static Long count = 0l;
    private Long id;
    private Integer numberOfNodes;
    private ArrayList<NeuralNode> neuralNodes;

    public NeuralLayer(Integer numberOfNodes) {
        this.id = ++NeuralLayer.count;
        this.numberOfNodes = numberOfNodes;
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
            NeuralNode node = new NeuralNode(this.id);
            neuralNodes.add(node);
        }
    }
}
