package cnn.execute;

import cnn.*;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.util.HashMap;

/**
 * Created by Tim on 4/1/2015.
 */
public class LeafCNN {

    public static void main(String[] args) throws Exception {
        new LeafCNN().run();
    }

    public void run() throws Exception {
        DoubleMatrix[][] images;
        DoubleMatrix labels;
        int patchDim = 3;
        int poolDim = 2;
        int numFeatures = 25;
        //int hiddenSize = 200;
        int imageRows = 80;
        int imageColumns = 60;
        double sparsityParam = 0.035;
        double lambda = 3e-3;
        double beta = 5;
        double alpha =  1e-4;
        int channels = 3;
        int hiddenSize = 200;
        int batchSize = 42;
        double momentum = 0;
        int iterations = 10000;

        String resFile = "Testing";

        ImageLoader loader = new ImageLoader();
        File folder = new File("C:\\Users\\jassmanntj\\Desktop\\CA-Leaves2");
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
        images = loader.getImgArr(0, batchSize);
        labels = loader.getLabels(0, batchSize);
        DoubleMatrix[][] testImages = loader.getTestArr(0, batchSize);
        DoubleMatrix testLabels = loader.getTestLabels(0, batchSize);
        ConvolutionLayer cl1 = new ConvolutionLayer(numFeatures, channels, patchDim);
        //ConvolutionLayer cl2 = new ConvolutionLayer(numFeatures, numFeatures, patchDim);
        //cl1.pretrain(images, numFeatures, 10);
        PoolingLayer cl3 = new PoolingLayer(poolDim);

        SparseAutoencoder sa = new SparseAutoencoder(numFeatures * ((imageRows-patchDim+1)/poolDim) * ((imageColumns - patchDim+1)/poolDim), hiddenSize, sparsityParam, lambda, beta, alpha);
        SparseAutoencoder sa2 = new SparseAutoencoder(hiddenSize, 100, sparsityParam, lambda, beta, alpha);

        SoftmaxClassifier sc = new SoftmaxClassifier(1e-4, hiddenSize, labels.columns);
        SparseAutoencoder[] saes = {sa};
        ConvPoolLayer[] cls = {cl1, cl3};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, sc, resFile);
        cnn.train(images, labels, testImages, testLabels, iterations, batchSize, momentum, alpha);
    }
}
