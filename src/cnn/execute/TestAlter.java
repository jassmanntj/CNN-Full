package cnn.execute;

import cnn.ImageLoader;
import cnn.Utils;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by jassmanntj on 4/30/2015.
 */
public class TestAlter {
    public static void main(String[] args) throws IOException {
        File folder = new File("C:\\Users\\jassmanntj\\Desktop\\CA-Leaves3");
        ImageLoader loader = new ImageLoader();
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        loader.loadFolder(folder, 3, 60, 80, labelMap);
        DoubleMatrix[][] images = loader.getImgArr(0, false, false, 0);
        DoubleMatrix[] image = images[0];
        DoubleMatrix[] image2 = new DoubleMatrix[images[0].length];
        DoubleMatrix[] image3 = new DoubleMatrix[images[0].length];
        DoubleMatrix[] image4 = new DoubleMatrix[images[0].length];
        DoubleMatrix[] image5 = new DoubleMatrix[images[0].length];
        DoubleMatrix[][] imgs = { image2, image3, image4, image5};
        for(int i = 0; i < imgs.length; i++) {
            for(int j = 0; j < imgs[i].length; j++) {
                imgs[i][j] = image[j].dup();
            }
        }
        Utils.alter(imgs);
        Utils.visualizeColorImg(image,"IMG");
        for(int i = 0; i < imgs.length; i++) {
            Utils.visualizeColorImg(imgs[i], "IMG"+i);
        }
    }
}
