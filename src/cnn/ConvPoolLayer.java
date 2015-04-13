package cnn;

import org.jblas.DoubleMatrix;

/**
 * Created by Tim on 4/1/2015.
 */
public abstract class ConvPoolLayer {
    public abstract DoubleMatrix[][] compute(DoubleMatrix[][] in);
    public DoubleMatrix[][] gradientCheck(CostResult cr, DoubleMatrix[][] in, DoubleMatrix labels, NeuralNetwork cnn) {
        return null;
    }
    public abstract DoubleMatrix[][] backPropagation(CostResult cr, double momentum, double alpha);
    public abstract CostResult cost(final DoubleMatrix[][] input, final DoubleMatrix[][] output, final DoubleMatrix delta[][]);
    public double getA() {
        return 0;
    }
}