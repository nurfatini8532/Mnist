package ai.skymind;

import java.io.File;
import java.util.Arrays;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

public class heart {
    private static File inputFile;
    private static int numLinesToSkip = 1;
    private static char delimiter = ',';

    private static int batchSize = 303;
    private static int labelIndex = 13;
    private static int numClasses = 2;

    public static void main(String[] args) throws Exception {
        // define csv file location
        inputFile = new ClassPathResource("heart.csv").getFile();

        RecordReader rr = new CSVRecordReader(numLinesToSkip,delimiter);
        rr.initialize(new FileSplit(inputFile));

        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();

        System.out.println(" Shape of all Data vector : ");
        System.out.println(Arrays.toString(allData.getFeatures().shape()));

        //split to training n test set
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        System.out.println(" Shape of training vector : ");
        System.out.println(Arrays.toString(trainingData.getFeatures().shape()));
        System.out.println(" Shape of test vector : ");
        System.out.println(Arrays.toString(testData.getFeatures().shape()));

        //create iterator
        DataSetIterator trainIterator = new ViewIterator(trainingData,4);
        DataSetIterator testIterator = new ViewIterator(testData,2);

        //normalization
        DataNormalization scaler = new NormalizerMinMaxScaler(0,1);
        scaler.fit(trainIterator);
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        System.out.println("Shape of training vector : ");
        System.out.println(Arrays.toString(trainIterator.next().getFeatures().shape()));
        System.out.println("Shape of test vector : ");
        System.out.println(Arrays.toString(testIterator.next().getFeatures().shape()));
        System.out.println("Training vector : ");
        System.out.println(trainIterator.next().getFeatures());
        System.out.println("Test vector : ");
        System.out.println(testIterator.next().getFeatures());

    }

}
