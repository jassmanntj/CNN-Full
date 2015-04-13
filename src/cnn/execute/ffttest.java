package cnn.execute;

/**
 * Created by jassmanntj on 4/13/2015.
 */
import org.jtransforms.fft.DoubleFFT_2D;

public class ffttest {
    public static void main(String[] args) {
        DoubleFFT_2D t = new DoubleFFT_2D(3,2);
        double[][] x = new double[3][4];
        x[0][0] = 1;
        x[0][1] = 2;
        x[1][0] = 3;
        x[1][1] = 4;
        x[2][0] = 5;
        x[2][1] = 6;
        t.realForwardFull(x);
        for(int i = 0; i < x.length; i++) {
            for(int j = 0; j < x[i].length; j++) {
                System.out.print(x[i][j]+" ");
            }
            System.out.println();
        }
    }
}
