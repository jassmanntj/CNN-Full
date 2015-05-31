package cnn.mobile;

import java.io.*;

import Jama.Matrix;

/**
 * Created by jassmanntj on 4/13/2015.
 */
public class DeviceFCLayer implements Serializable {
    private int activation;
    private double a;
    private Matrix theta;
    private Matrix bias;

    public DeviceFCLayer(int activation, double a, Matrix theta, Matrix bias) {
        this.activation = activation;
        this.a = a;
        this.theta = theta;
        this.bias = bias;
    }

    public Matrix compute(Matrix input) {
        Matrix result = input.times(theta);
        if(bias != null) result.plusEquals(bias);
        return JDeviceUtils.activationFunction(activation, result, a);
    }
}
