package cnn.execute;

import cnn.*;
import org.jblas.DoubleMatrix;

/**
 * Created by Tim on 3/31/2015.
 */
public class LeafCNNGradientChecking {

    public static void main(String[] args) throws Exception {
        new LeafCNNGradientChecking().run();
    }

    public void run() throws Exception {
        DoubleMatrix[][] images;
        DoubleMatrix labels;
        int patchDim = 5;
        int poolDim = 1;
        int numFeatures = 3;
        //int hiddenSize = 200;
        int imageRows = 16;
        int imageColumns = 12;
        double sparsityParam = 0.035;
        double lambda = 3e-3;
        double beta = 5;
        double alpha =  1e-2;
        int channels = 3;
        int hiddenSize = 150;
        int batchSize = 30;
        double momentum = 0;
        double dropout = 0;

        String resFile = "Testing";
        images = new DoubleMatrix[3][channels];
        for(int i = 0; i < channels; i++) {
            images[0][i] = DoubleMatrix.rand(imageRows, imageColumns);
        }
        for(int i = 0; i < channels; i++) {
            images[1][i] = DoubleMatrix.rand(imageRows, imageColumns);
        }
        for(int i = 0; i < channels; i++) {
            images[2][i] = DoubleMatrix.rand(imageRows, imageColumns);
        }
        labels = new DoubleMatrix(3,2);
        labels.put(0,0,1);
        labels.put(1,1,1);
        labels.put(2,1,1);
        //ImageLoader loader = new ImageLoader();
        //File folder = new File("C:\\Users\\Tim\\Desktop\\CA-Leaves2");
        //HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        //loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
        //images = loader.getImgArr();
        //labels = loader.getLabels();
        int cl = 2;

        ConvolutionLayer cl1 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, dropout, Utils.PRELU);
        ConvolutionLayer cl2 = new ConvolutionLayer(numFeatures, numFeatures, patchDim, lambda, dropout, Utils.PRELU);
        PoolingLayer cl3 = new PoolingLayer(poolDim, PoolingLayer.MEAN);

        FCLayer sa = new FCLayer(numFeatures * ((imageRows-cl*patchDim+cl)/poolDim) * ((imageColumns - cl*patchDim+cl)/poolDim), hiddenSize, lambda, dropout, Utils.PRELU);
        FCLayer sa2 = new FCLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.PRELU);

        SoftmaxClassifier sc = new SoftmaxClassifier(1e-4, hiddenSize, labels.columns);
        FCLayer[] saes = {sa, sa2};
        ConvPoolLayer[] cls = {cl1, cl2};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, sc, resFile);
        while(true) {
            cnn.train(images, labels, images, labels, 1, 2, momentum, alpha, 1);
            cnn.gradientCheck(images, labels);
        }
    }
}
