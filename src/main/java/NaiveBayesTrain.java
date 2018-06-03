package main.java;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NaiveBayesTrain {

    public static void main(String ... args){
        SparkSession sparkSession = SparkSession.builder().appName("My App").master("local[2]").getOrCreate();
       //Load preprocessed training data
        Dataset<String> dataset = sparkSession.read().textFile("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\processed.csv");

        dataset.printSchema();
        dataset.show(10);
        List<String> dataSetString = dataset.collectAsList();
        List<Row> rowList = new ArrayList<>();

        for(String string:dataSetString){
            int label = Integer.parseInt(string.split(",")[0]);
            String content = string.split(",")[1].substring(1);
            rowList.add(RowFactory.create(label,content));
        }

        //Create schema for tweets
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("tweet", DataTypes.StringType, false, Metadata.empty())
        });


        Dataset <Row> formattedDataSet = sparkSession.createDataFrame(rowList,schema);

        formattedDataSet.printSchema();
        formattedDataSet.show(10);

        Tokenizer tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("tokens");

        Dataset<Row> tokenizedData = tokenizer.transform(formattedDataSet);

        tokenizedData.printSchema();
        tokenizedData.show(10);

        CountVectorizer countVectorizer = new CountVectorizer().setInputCol("tokens").setOutputCol("rawFeatures");
        CountVectorizerModel countVectorizerModel = countVectorizer.fit(tokenizedData);

        Dataset <Row> countVectorizerFeatures = countVectorizerModel.transform(tokenizedData);

        countVectorizerFeatures.show(10);

        IDFModel idfModel = new IDF().setInputCol("rawFeatures").setOutputCol("features").fit(countVectorizerFeatures);

        Dataset <Row> idfFeatures = idfModel.transform(countVectorizerFeatures);

        idfFeatures.show(10);



// Split the data into train and test
        Dataset<Row>[] splits = idfFeatures.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

// create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();

// train the model
        NaiveBayesModel model = new NaiveBayes().setSmoothing(1.0).fit(train);

// Select example rows to display.
        Dataset<Row> predictions = model.transform(test);
        predictions.show();
        predictions.printSchema();
// compute accuracy on the test set
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("\n\nTEST SET ACCURACY = " + accuracy);



        // predictions.coalesce(1).rdd().saveAsTextFile("/media/banqo/Banqo/Dev/Predictions");
        //   predictions.write().csv("/media/banqo/Banqo/Dev/PredictionsCSV");
//        try {
//            model.save("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\NAIVEBAYESMODEL");
//            countVectorizerModel.save("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\VectorizerModel");
//            idfModel.save("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\IDFModel");
//
//        }
//        catch (IOException e){
//            System.out.println(e.getMessage());
//        }
        sparkSession.stop();
    }



    }

