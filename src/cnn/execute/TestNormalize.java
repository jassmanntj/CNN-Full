package cnn.execute;

import cnn.ImageLoader;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by jassmanntj on 4/2/2015.
 */
public class TestNormalize {

    public static void main(String[] args) throws IOException {
        ImageLoader loader = new ImageLoader();
        File folder = new File("C:\\Users\\jassmanntj\\Desktop\\CA-Leaves2");
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        loader.loadFolder(folder, 3, 60, 80, labelMap);
        DoubleMatrix[][] images = loader.getImgArr(0, 30);
        DoubleMatrix imgs = loader.getImgs();
        System.out.println(images[0][0]);
        System.out.println(imgs);
    }
}
