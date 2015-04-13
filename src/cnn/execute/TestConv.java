package cnn.execute;

import cnn.Utils;
import org.jblas.DoubleMatrix;

/**
 * Created by jassmanntj on 4/12/2015.
 */
public class TestConv {
    public static void main(String[] args) {
        DoubleMatrix a = DoubleMatrix.ones(2, 4);
        DoubleMatrix b = DoubleMatrix.zeros(2, 2);
        b.put(0,0,1);
        b.put(0,1,1);
        a.put(0,0,0);
        /*for(int i = 0; i < a.rows; i++) {
            System.out.println(a.getRow(i));
        }
        System.out.println("--------------");
        for(int i = 0; i < b.rows; i++) {
            System.out.println(b.getRow(i));
        }
        System.out.println("--------------");
*/
        DoubleMatrix res = Utils.conv2d(a,b,true);
        for(int i = 0; i < res.rows; i++) {
            System.out.println(res.getRow(i));
        }
    }
}
