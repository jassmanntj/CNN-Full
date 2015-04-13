package cnn;

import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


/**
 * Created by jassmanntj on 3/25/2015.
 */
public class ConvolutionLayer extends ConvPoolLayer {
    private DoubleMatrix theta[][];
    private DoubleMatrix thetaVelocity[][];
    private double a;
    private double aVelocity;
    private DoubleMatrix bias;
    private DoubleMatrix biasVelocity;
    private int patchDim;
    private int layer1 = Utils.PRELU;

    public ConvolutionLayer(int numFeatures, int channels, int patchDim) {
        this.patchDim = patchDim;

        bias = new DoubleMatrix(numFeatures);
        biasVelocity = new DoubleMatrix(numFeatures);
        theta = new DoubleMatrix[numFeatures][channels];
        thetaVelocity = new DoubleMatrix[numFeatures][channels];

        for(int i = 0; i < numFeatures; i++) {
            for(int j = 0; j < channels; j++) {
                theta[i][j] = initializeTheta(patchDim, numFeatures);
                thetaVelocity[i][j] = new DoubleMatrix(patchDim, patchDim);
            }
        }
        a = .25;
    }

    public void pretrain(DoubleMatrix[][] images, int numFeatures, int iterations) {
        LinearDecoder ae = new LinearDecoder(patchDim, theta[0].length, numFeatures, 0.035, 3e-3, 5, 1e-3);
        DoubleMatrix patches = ImageLoader.sample(patchDim, patchDim, 10000, images);
        ae.train(patches, patches, iterations);
        this.theta = ae.getThetaArr();
        this.bias = ae.getBias();
    }

    private DoubleMatrix initializeTheta(int patchDim, int numFeatures) {
        double stdev = Math.sqrt(2.0/(patchDim*patchDim*numFeatures));
        DoubleMatrix res = DoubleMatrix.randn(patchDim, patchDim);
        res.muli(stdev);
        return res;
    }

    public DoubleMatrix[][] compute(final DoubleMatrix[][] input) {
        final DoubleMatrix[][] result = new DoubleMatrix[input.length][theta.length];

        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < theta.length; feature++) {
                    DoubleMatrix res = new DoubleMatrix(input[imageNum][0].rows-theta[0][0].rows+1, input[imageNum][0].columns-theta[0][0].columns+1);
                    for(int channel = 0; channel < theta[feature].length; channel++) {
                        res.addi(Utils.conv2d(input[imageNum][channel], theta[feature][channel], true));
                    }
                    result[imageNum][feature] = Utils.activationFunction(layer1, res.add(bias.get(feature)), a);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return result;
    }


    public DoubleMatrix[][] gradientCheck(CostResult cr, DoubleMatrix[][] in, DoubleMatrix labels, NeuralNetwork cnn) {
        double epsilon = 0.0001;
        DoubleMatrix biasG = new DoubleMatrix(bias.length);
        for(int i = 0; i < bias.length; i++) {
            bias.put(i, bias.get(i)+epsilon);
            CostResult costPlus = cnn.computeCost(in, labels);
            bias.put(i, bias.get(i)-2*epsilon);
            CostResult costMinus = cnn.computeCost(in, labels);
            bias.put(i, bias.get(i)+epsilon);
            biasG.put(i, (costPlus.cost-costMinus.cost)/(2*epsilon));
        }
        DoubleMatrix biasA = biasG.add(cr.biasGrad);
        DoubleMatrix biasS = biasG.sub(cr.biasGrad);
        System.out.println("CL Bias Diff: "+biasS.norm2()/biasA.norm2());

        for(int i = 0; i < theta.length; i++) {
            for(int j = 0; j < theta[i].length; j++) {
                DoubleMatrix thetaG = new DoubleMatrix(cr.tGrad[i][j].rows, cr.tGrad[i][j].columns);
                for(int k = 0; k < theta[i][j].length; k++) {
                    theta[i][j].put(k, theta[i][j].get(k)+epsilon);
                    CostResult costPlus = cnn.computeCost(in, labels);
                    theta[i][j].put(k, theta[i][j].get(k)-2*epsilon);
                    CostResult costMinus = cnn.computeCost(in, labels);
                    theta[i][j].put(k, theta[i][j].get(k)+epsilon);
                    thetaG.put(k, (costPlus.cost-costMinus.cost)/(2*epsilon));
                }
                DoubleMatrix thetaA = thetaG.add(cr.tGrad[i][j]);
                DoubleMatrix thetaS = thetaG.sub(cr.tGrad[i][j]);
                System.out.println("CL Theta "+i+"/"+theta.length+":"+j+"/"+theta[i].length+" Diff: "+thetaS.norm2()/thetaA.norm2());
            }
        }

        a += epsilon;
        CostResult costP = cnn.computeCost(in, labels);
        a -= 2*epsilon;
        CostResult costM = cnn.computeCost(in, labels);
        a += epsilon;
        double aG = (costP.cost-costM.cost)/(2*epsilon);
        System.out.println("CL a: "+Math.abs((cr.aGrad-aG)/(cr.aGrad+aG)));
        return cr.delt;
    }

    public CostResult cost(final DoubleMatrix[][] input, final DoubleMatrix[][] output, final DoubleMatrix delta[][]) {
        final DoubleMatrix[][] delt = new DoubleMatrix[input.length][theta[0].length];
        final DoubleMatrix[][] thetaGrad = new DoubleMatrix[theta.length][theta[0].length];
        //System.out.println(delta[0][0]);
        for(int image = 0; image < input.length; image++) {
            for (int channel = 0; channel < theta[0].length; channel++) {
                delt[image][channel] = new DoubleMatrix(input[0][0].rows, input[0][0].columns);
            }
        }
        for(int feature = 0; feature < theta.length; feature++) {
            for (int channel = 0; channel < theta[0].length; channel++) {
                thetaGrad[feature][channel] = new DoubleMatrix(patchDim, patchDim);
            }
        }
        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < theta.length; feature++) {
                    delta[imageNum][feature].muli(Utils.activationGradient(layer1, output[imageNum][feature], a));
                    for(int channel = 0; channel < theta[feature].length; channel++) {
                        delt[imageNum][channel].addi(Utils.conv2d(delta[imageNum][feature], Utils.reverseMatrix(theta[feature][channel]), false));
                        thetaGrad[feature][channel].addi(Utils.conv2d(input[imageNum][channel], delta[imageNum][feature], true).div(input.length));
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        DoubleMatrix bGrad = new DoubleMatrix(bias.length);
        for(int i = 0; i < theta.length; i++) {
            double deltMean = 0;
            for(int j = 0; j < input.length; j++) {
                deltMean += delta[j][i].sum();
            }
            bGrad.put(i, deltMean / input.length);
        }
        //System.out.println(delta[0][0]);
        double aGrad = Utils.aGrad(layer1, output, delta, a);
        return new CostResult(0, thetaGrad, bGrad, delt, aGrad);
    }

    public DoubleMatrix[][] backPropagation(CostResult cr, double momentum, double alpha) {
        biasVelocity.muli(momentum).addi(cr.biasGrad.mul(alpha));
        bias.subi(biasVelocity);
        for(int i = 0; i < theta.length; i++) {
            for(int j = 0; j < theta[i].length; j++) {
                thetaVelocity[i][j].muli(momentum).addi(cr.tGrad[i][j].mul(alpha));
                theta[i][j].subi(thetaVelocity[i][j]);
            }
        }
        aVelocity = aVelocity * momentum + cr.aGrad * alpha;
        a -= aVelocity;//cr.aGrad * alpha;
        //System.out.println("CL A: "+a);
        return cr.delt;
    }

    public double getA() {
        return a;
    }

    public void writeLayer(String filename) {
        /*try {
            FileWriter fw = new FileWriter(filename);
            BufferedWriter writer = new BufferedWriter(fw);
            writer.write(theta.length+","+theta.columns+"\n");
            for(int i = 0; i < theta.rows; i++) {
                for(int j = 0; j < theta.columns; j++) {
                    writer.write(theta.get(i,j)+",");
                }
            }
            writer.write("\n"+bias.rows+","+bias.columns+"\n");
            for(int i = 0; i < bias.rows; i++) {
                for(int j = 0; j < bias.columns; j++) {
                    writer.write(bias.get(i,j)+",");
                }
            }
            writer.write("\n"+imageRows+"\n"+imageCols+"\n"+patchDim+"\n"+poolDim);
            writer.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }*/
    }

    public void loadLayer(String filename) {
        /*try {
            FileReader fr = new FileReader(filename);
            @SuppressWarnings("resource")
            BufferedReader reader = new BufferedReader(fr);
            String[] line = reader.readLine().split(",");
            theta = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
            line = reader.readLine().split(",");
            for(int i = 0; i < theta.rows; i++) {
                for(int j = 0; j < theta.columns; j++) {
                    theta.put(i, j, Double.parseDouble(line[i * theta.columns + j]));
                }
            }
            line = reader.readLine().split(",");
            bias = new DoubleMatrix(Integer.parseInt(line[0]), Integer.parseInt(line[1]));
            line = reader.readLine().split(",");
            for(int i = 0; i < bias.rows; i++) {
                for(int j = 0; j < bias.columns; j++) {
                    bias.put(i, j, Double.parseDouble(line[i * bias.columns + j]));
                }
            }
            imageRows = Integer.parseInt(reader.readLine());
            imageCols = Integer.parseInt(reader.readLine());
            patchDim = Integer.parseInt(reader.readLine());
            poolDim = Integer.parseInt(reader.readLine());
            reader.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }*/
    }
}
