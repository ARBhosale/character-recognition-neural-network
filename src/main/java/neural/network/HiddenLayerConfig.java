package neural.network;

public class HiddenLayerConfig {
    private Integer numberOfHiddenUnits;
    private char transformFunction;
    private Double threshold = 0D;

    public Double getThreshold() {
        return threshold;
    }

    public void setThreshold(Double threshold) {
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
}
