package neural.network;

public class HiddenLayerConfig {
    private Integer numberOfHiddenUnits;

    public HiddenLayerConfig(Integer numberOfHiddenUnits) {
        this.numberOfHiddenUnits = numberOfHiddenUnits;
    }

    public Integer getNumberOfHiddenUnits() {
        return numberOfHiddenUnits;
    }

    public void setNumberOfHiddenUnits(Integer numberOfHiddenUnits) {
        this.numberOfHiddenUnits = numberOfHiddenUnits;
    }
}
