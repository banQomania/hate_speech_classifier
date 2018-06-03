package main.java;


import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NaiveBayesPredict {

    public static class Prediction{
        private String id;
        private String label;

        public Prediction(String id, String label) {
            this.id = id;
            this.label = label;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getLabel() {
            return label;
        }

        public void setLabel(String label) {
            this.label = label;
        }
    }


    public static void main(String ... args){
        SparkSession sparkSession = SparkSession.builder().appName("My App").master("local[4]").getOrCreate();

        Dataset<String> dataset = sparkSession.read().textFile("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\test_processed.csv");

        dataset.show();
        List<String> dataSetString = dataset.collectAsList();
        System.out.print(dataSetString.size());
        List<Row> rowList = new ArrayList<>();

        for(String string:dataSetString){
            int id = Integer.parseInt(string.split(",")[0]);
            String content = string.split(",")[1].substring(1);
            rowList.add(RowFactory.create(id,content));
        }


        StructType schema = new StructType(new StructField[]{
                new StructField("id",DataTypes.IntegerType,false,Metadata.empty()),
                new StructField("tweet", DataTypes.StringType, false, Metadata.empty())
        });
        Dataset<Row> formattedDataSet = sparkSession.createDataFrame(rowList,schema);

        formattedDataSet.printSchema();
        formattedDataSet.show(10);

        Tokenizer tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("tokens");

        Dataset<Row> tokenizedData = tokenizer.transform(formattedDataSet);

        tokenizedData.printSchema();
        tokenizedData.show(10);

        //Load the CountVectorizer Model which contains learnt vocabulary from the training set
        CountVectorizerModel countVectorizerModel = CountVectorizerModel.load("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\VectorizerModel");
        //Transform the unlabelled tweets to get raw count features
        Dataset <Row> countVectorizerFeatures = countVectorizerModel.transform(tokenizedData);
        countVectorizerFeatures.show();

        //Load the Inverse Document Frequency model adapted to the training set

        IDFModel idfModel = IDFModel.load("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\IDFModel");
        //Calculate IDF use CountVectorizer features to get features going to be used for training
        Dataset <Row> idfFeatures = idfModel.transform(countVectorizerFeatures);

        idfFeatures.show();
        //Load the weights of the NaiveBayesModel
        NaiveBayesModel naiveBayesModel = NaiveBayesModel.load("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\NAIVEBAYESMODEL");
        // Predict the the class of the tweet
        Dataset<Row> prediction = naiveBayesModel.transform(idfFeatures);
        prediction.printSchema();
        prediction.show();



//Select columns needed for classification and output them to a csv file
        prediction.select("id","prediction").coalesce(1).write().option("header","false")
                .mode("overwrite")
                .csv("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\test_predictions.csv");

        sparkSession.stop();

    }
}
