package neural.network;

public class HiddenLayerConfig {
    private Integer numberOfHiddenUnits;
    private char transformFunction;
    private double threshold = 0D;

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public char getTransformFunction() {
        return transformFunction;
    }

    public void setTransformFunction(char transformFunction) {
        this.transformFunction = transformFunction;
    }

    public Integer getNumberOfHiddenUnits() {
        return numberOfHiddenUnits;
    }

    public void setNumberOfHiddenUnits(Integer numberOfHiddenUnits) {
        this.numberOfHiddenUnits = numberOfHiddenUnits;
    }

    public HiddenLayerConfig(Integer numberOfHiddenUnits, char transformFunction) {
        this.numberOfHiddenUnits = numberOfHiddenUnits;
        this.transformFunction = transformFunction;
    }

    @Override
    public String toString() {
        return "NumberOfHiddenUnits=" + numberOfHiddenUnits +
                "\nTransformFunction=" + transformFunction +
                "\nThreshold=" + threshold;
    }
}
