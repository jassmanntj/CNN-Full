package cnn;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.Singular;
import org.jtransforms.fft.DoubleFFT_2D;

public class Utils {
    public static final int NUMTHREADS = 8;
    public static final int NONE = 0;
    public static final int SIGMOID = 1;
    public static final int PRELU = 2;
    public static final int RELU = 3;
    public static final int SOFTMAX = 4;
    public static final int CONVLAYER = 0;
    public static final int POOLLAYER = 1;

	/*public static DoubleMatrix ZCAWhiten(DoubleMatrix input, double epsilon) {
        input.subiColumnVector(input.rowMeans());
        DoubleMatrix sigma = DoubleMatrix.diag(DoubleMatrix.diag(input.transpose().mmul(input)));
		sigma.divi(input.rows);
		DoubleMatrix[] svd = Singular.fullSVD(sigma);
		DoubleMatrix ZCAWhite = svd[1];
		ZCAWhite.addi(epsilon);
        MatrixFunctions.sqrti(ZCAWhite);
		ZCAWhite.rdivi(1);
		ZCAWhite = DoubleMatrix.diag(ZCAWhite);
		ZCAWhite = svd[0].mmul(ZCAWhite).mmul(svd[0].transpose());
		return ZCAWhite.mmul(input);
	}*/

    public static DoubleMatrix[][] ZCAWhiten(DoubleMatrix[][] input, double epsilon) {
        DoubleMatrix img = flatten(input);
        DoubleMatrix mean = img.rowMeans();
        img.subiColumnVector(mean);
        DoubleMatrix sigma = img.mul(img).rowMeans();
        sigma = DoubleMatrix.diag(sigma);
        DoubleMatrix[] svd = Singular.fullSVD(sigma);
        DoubleMatrix s = DoubleMatrix.diag(MatrixFunctions.sqrt(svd[1].add(epsilon)).rdiv(1));
        DoubleMatrix res = svd[0].mmul(s).mmul(svd[0].transpose()).mmul(img);
        return expand(res, input[0].length, input[0][0].rows, input[0][0].columns);
    }

    public static DoubleMatrix[][] expand(DoubleMatrix in, int channels, int rows, int cols) {
        DoubleMatrix[][] result = new DoubleMatrix[in.rows][channels];
        for(int i = 0; i < in.rows; i++) {
            for(int j = 0; j < channels; j++) {
                result[i][j] = new DoubleMatrix(rows, cols);
                for(int k = 0; k < rows; k++) {
                    for(int l = 0; l < cols; l++) {
                        result[i][j].put(k, l, in.get(i, j * rows * cols + k * cols + l));
                    }
                }
            }
        }
        return result;
    }

	public static DoubleMatrix conv2d(DoubleMatrix input, DoubleMatrix kernel, boolean valid) {
		int inputRows = input.rows;
		int inputCols = input.columns;
		int kernelRows = kernel.rows;
		int kernelCols = kernel.columns;
		int totalRows = inputRows + kernelRows - 1;
		int totalCols = inputCols + kernelCols - 1;
		kernel = reverseMatrix(kernel);
		input = DoubleMatrix.concatHorizontally(input, DoubleMatrix.zeros(input.rows, kernel.columns-1));
		input = DoubleMatrix.concatVertically(input, DoubleMatrix.zeros(kernel.rows-1, input.columns));
		kernel = DoubleMatrix.concatHorizontally(kernel, DoubleMatrix.zeros(kernel.rows, input.columns-kernel.columns));
		kernel = DoubleMatrix.concatVertically(kernel, DoubleMatrix.zeros(input.rows-kernel.rows,kernel.columns));
		ComplexDoubleMatrix inputDFT = new ComplexDoubleMatrix(input);
		ComplexDoubleMatrix kernelDFT = new ComplexDoubleMatrix(kernel);
		DoubleFFT_2D t = new DoubleFFT_2D(inputDFT.columns, inputDFT.rows);
		t.complexForward(inputDFT.data);
		t.complexForward(kernelDFT.data);
		kernelDFT.muli(inputDFT);
		t.complexInverse(kernelDFT.data, true);
		int rowSize = inputRows - kernelRows + 1;
		int colSize = inputCols - kernelCols + 1;
		DoubleMatrix result = kernelDFT.getReal();
        if(!valid) return result;
        else {
            int startRows = (totalRows - rowSize) / 2;
            int startCols = (totalCols - colSize) / 2;
            result = result.getRange(startRows, startRows + rowSize, startCols, startCols + colSize);
            return result;
        }
	}
	
	public static DoubleMatrix reverseMatrix(DoubleMatrix mat) {
		mat = flipHorizontal(mat);
		return flipVerticali(mat);
	}

    public static DoubleMatrix flipHorizontal(DoubleMatrix mat) {
        mat = mat.dup();
        return flipHorizontali(mat);
    }

    public static DoubleMatrix flipVertical(DoubleMatrix mat) {
        mat = mat.dup();
        return flipVerticali(mat);
    }

    public static DoubleMatrix flipHorizontali(DoubleMatrix mat) {
        for(int i = 0; i < mat.rows/2; i++) {
            mat.swapRows(i, mat.rows-i-1);
        }
        return mat;
    }

    public static DoubleMatrix flipVerticali(DoubleMatrix mat) {
        for(int i = 0; i < mat.columns/2; i++) {
            mat.swapColumns(i, mat.columns-i-1);
        }
        return mat;
    }
	
	public static void visualizeColor(int width, int height, int images, DoubleMatrix img, String filename) throws IOException {
		BufferedImage image = new BufferedImage(width*images+images*2+2, height*images+images*2+2, BufferedImage.TYPE_INT_RGB);
		DoubleMatrix tht1 = img.dup();
		tht1.subi(tht1.min());
		tht1.divi(tht1.max());
		tht1.muli(255);
		for(int k = 0; k < images; k++) {
			for(int l = 0; l < images; l++) {
				if(k*images+l < tht1.rows) { 
					DoubleMatrix row = tht1.getRow(k*images+l);
					int channelSize = row.length/3;
					double[] r = new double[channelSize];
					double[] g = new double[channelSize];
					double[] b = new double[channelSize];
					System.arraycopy(row.data, 0, r, 0, channelSize);
					System.arraycopy(row.data, channelSize, g, 0, channelSize);
					System.arraycopy(row.data, 2*channelSize, b, 0, channelSize);
					for(int i = 0; i < height; i++) {
						for(int j = 0; j < width; j++) {
							int col = ((int)r[i*width+j] << 16) | ((int)g[i*width+j] << 8) | (int)b[i*width+j];
							image.setRGB(l*(width+2)+2+j, k*(height+2)+2+i, col);
						}
					}
				}
			}
		}
		File imageFile = new File(filename);
		ImageIO.write(image, "png", imageFile);
	}
	
	public static void visualize(int size, int images, DoubleMatrix input, String filename) throws IOException {
		BufferedImage image = new BufferedImage((size+2)*images+2, (size+2)*images+2, BufferedImage.TYPE_INT_RGB);
		DoubleMatrix tht1 = input.dup();
		tht1.subi(tht1.min());
		tht1.divi(tht1.max());
		tht1.muli(255);
		System.out.println(tht1.getRow(0));
		for(int k = 0; k < images; k++) {
			for(int l = 0; l < images; l++) {
				for(int i = 0; i < size; i++) {
					for(int j = 0; j < size; j++) {
						int imageNo = (int)(Math.random() * 9999);
						int val = (int)tht1.get(imageNo,i*size+j);
						int col = (val << 16) | (val << 8) | val;
						image.setRGB(l*(2+size)+2+j, k*(2+size)+2+i, col);
					}
				}
			}
		}
		File imageFile = new File(filename);
		ImageIO.write(image, "png", imageFile);
	}

    public static void visualizeColorImg(DoubleMatrix[] img, String filename) throws IOException {
        int width = img[0].columns;
        int height = img[0].rows;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        double min = 255;
        double max = -255;
        for(int i = 0; i < img.length; i++) {
            min = img[i].min() < min ? img[i].min():min;
            max = img[i].max() > max ? img[i].max():max;
        }
        max = max - min;
        for(int i = 0; i < img.length; i++) {
            img[i].subi(min);
            img[i].divi(max);
            img[i].muli(255);
        }
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                int col = ((int)img[0].get(i,j) << 16) | ((int)img[1].get(i,j) << 8) | (int)img[2].get(i,j);
                image.setRGB(j,i, col);
            }
        }
        File imageFile = new File(filename+".png");
        ImageIO.write(image, "png", imageFile);
    }
	
	public static DoubleMatrix sigmoid(DoubleMatrix z) {
		return MatrixFunctions.exp(z.neg()).add(1).rdiv(1);
	}
	
	public static DoubleMatrix sigmoidGradient(DoubleMatrix a) {
		return a.rsub(1).mul(a);
	}
	
	public static int[][] computeResults(DoubleMatrix result) {
		int[][] results = new int[result.rows][result.columns];
		for(int i = 0; i < result.rows; i++) {
            double[] current = new double[result.columns];
			for(int j = 0; j < result.columns; j++) {
                for(int k = 0; k < result.columns; k++) {
                    if(result.get(i,j) > current[k]) {
                        for(int l = result.columns-1; l > k; l--) {
                            current[l] = current[l-1];
                            results[i][l] = results[i][l-1];
                        }
                        current[k] = result.get(i,j);
                        results[i][k] = j;
                        break;
                    }
                }
			}
		}
		return results;
	}

    public static DoubleMatrix flatten(DoubleMatrix[][] z) {
        DoubleMatrix images = null;
        for(int i = 0; i < z.length; i++) {
            DoubleMatrix image = null;
            for(int j = 0; j < z[i].length; j++) {
                for(int k = 0; k < z[i][j].rows; k++) {
                    if(image == null) image = z[i][j].getRow(k);
                    else image = DoubleMatrix.concatHorizontally(image, z[i][j].getRow(k));
                }
            }
            if(images == null) images = image;
            else images = DoubleMatrix.concatVertically(images, image);
        }
        return images;
    }

    public static double aGrad(int type, DoubleMatrix result, DoubleMatrix delta) {
        double aGrad = 0;
        switch(type) {
            case PRELU:
                for(int i = 0; i < result.length; i++) {
                    if( result.get(i) <= 0) aGrad += result.get(i)*delta.get(i);
                }
                return aGrad/result.rows;
            case SIGMOID: return 0;
            case RELU: return 0;
            case NONE: return 0;
            default: return 0;
        }
    }

    public static void printMatrix(DoubleMatrix mat, BufferedWriter writer) throws IOException {
        for(int i = 0; i < mat.rows; i++) {
            for(int j = 0; j < mat.columns; j++) {
                writer.write(mat.get(i,j)+" ");
            }
            writer.write("\n");
        }
        writer.write("\n");
    }

    public static void alter(DoubleMatrix[][] images) {
        class AlterThread implements Runnable {
            private DoubleMatrix[] image;

            public AlterThread(DoubleMatrix[] image) {
                this.image = image;
            }

            @Override
            public void run() {
                DoubleMatrix a = DoubleMatrix.randn(image.length).mul(0.1);
                DoubleMatrix i = new DoubleMatrix(image.length, image[0].length);
                for(int k = 0; k < image.length; k++) {
                    for (int j = 0; j < image[0].length; j++) {
                        i.put(k, j, image[k].get(j));
                    }
                }
                i.subiColumnVector(i.rowMeans());
                i = i.mmul(i.transpose()).div(i.columns);
                DoubleMatrix[] svd = Singular.fullSVD(i);
                svd[1].muli(a);
                DoubleMatrix res = svd[0].mmul(svd[1]);
                for(int j = 0; j < image.length; j++) {
                    image[j].addi(res.get(j));
                }
                if(Math.random() > 0.5) {
                    for (int j = 0; j < image.length; j++) {
                        image[j] = Utils.flipHorizontal(image[j]);
                    }
                }
                if(Math.random() > 0.5) {
                    for (int j = 0; j < image.length; j++) {
                        image[j] = Utils.flipVertical(image[j]);
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(DoubleMatrix[] image : images) {
            Runnable worker = new AlterThread(image);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
    }

    public static double aGrad(int type, DoubleMatrix[][] result, DoubleMatrix[][] delta) {
        double aGrad = 0;
        switch(type) {
            case PRELU:
                for(int i = 0; i < result.length; i++) {
                    for(int j = 0; j < result[i].length; j++) {
                        for(int k = 0; k < result[i][j].length; k++) {
                            if(result[i][j].get(k) <= 0) aGrad += result[i][j].get(k) * delta[i][j].get(k);
                        }
                    }
                }
                return aGrad/result.length;
            case SIGMOID:
            case RELU:
            case NONE:
            default: return aGrad;
        }
    }

    public static DoubleMatrix activationFunction(int type, DoubleMatrix z, double a) {
        switch(type) {
            case SIGMOID:
                return sigmoid(z);
            case PRELU:
                return prelu(z, a);
            case RELU:
                return relu(z);
            case SOFTMAX:
                return softmax(z);
            case NONE:
                return z;
            default:
                return sigmoid(z);
        }
    }

    private static DoubleMatrix softmax(DoubleMatrix z) {
        DoubleMatrix p = z.subColumnVector(z.rowMaxs());
        MatrixFunctions.expi(p);
        p.diviColumnVector(p.rowSums());
        return p;
    }

    public static DoubleMatrix activationGradient(int type, DoubleMatrix z, double a) {
        switch(type) {
            case SIGMOID:
                return sigmoidGradient(z);
            case PRELU:
                return preluGradient(z, a);
            case RELU:
                return reluGradient(z);
            case NONE:
                return DoubleMatrix.ones(z.rows, z.columns);
            default:
                return sigmoidGradient(sigmoid(z));
        }
    }

    public static DoubleMatrix prelu(DoubleMatrix z, double a) {
        //return z.le(0).mul(a-1).add(1).mul(z);
        DoubleMatrix res = z.dup();
        for(int i = 0; i < res.rows; i++) {
            for(int j = 0; j < res.columns; j++) {
                double k = res.get(i,j);
                res.put(i,j,Math.max(0,k)+a*Math.min(0,k));
            }
        }
        return res;
    }

    public static DoubleMatrix relu(DoubleMatrix z) {
        return prelu(z, 0);
    }

    public static DoubleMatrix preluGradient(DoubleMatrix z, double a) {
        //return z.le(0).mul(a-1).add(1);
        DoubleMatrix res = new DoubleMatrix(z.rows, z.columns);
        for(int i = 0; i < z.rows; i++) {
            for(int j = 0; j < z.columns; j++) {
                double k = z.get(i,j);
                res.put(i,j, k>0? 1:a);
            }
        }
        return res;
    }

    public static DoubleMatrix reluGradient(DoubleMatrix z) {
        return z.gt(0);
    }


	
}
