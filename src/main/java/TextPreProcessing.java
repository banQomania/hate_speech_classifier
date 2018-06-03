package main.java;

import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import java.io.*;
import java.util.*;

public  class TextPreProcessing {

    private static List<Model> modelList = new ArrayList<>();
    private static List<Model> tokenizedModels = new ArrayList<>();
    private static File stopWordsFile = new File("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\stopwords.txt");
    private static Set<String> stopWords = new HashSet<>();
    private static List<Model> stopWordRemoved = new ArrayList<>();
    private static List<String> stopWordRemovedTokens = new ArrayList<>();
    private static List<Model> filteredModels = new ArrayList<>();
    private static List<Model> stemmedDocuments = new ArrayList<>();
    private static Boolean isTrain = true;

    public TextPreProcessing() {
    }

    public static void main(String ... args){

        //Change to denote whether you are preprocessing training or test data
        //isTrain = false;

        readDataSet();
       readTestDataSet();
      tokenizeTweets();
        System.out.println(modelList.size());
        readStopWords();
        removeStopWords();
        filterModels();
        stemTokens();
        savePreProcessedCSV();
    }

    private static void savePreProcessedCSV() {
        String trainDataSetPath ="";
        if(isTrain){
            trainDataSetPath ="C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\processed.csv";
        }
        else {
            trainDataSetPath ="C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\test_processed.csv";
        }

        try(FileWriter fileWriter = new FileWriter(trainDataSetPath)){

            for(Model model: stemmedDocuments){
                if(isTrain) {
                    fileWriter.write(model.getLabel() + ", " + model.toString() + "\n");
                }
                else if(!isTrain){
                    fileWriter.write(model.getId() + ", " + model.toString() + "\n");
                }
            }

            fileWriter.flush();
            fileWriter.close();
        }catch (IOException e){
            System.out.println(e.getMessage());
        }

    }

    public static void readTestDataSet(){
        BufferedReader bufferedReader = null;

        String line;
        try{
            bufferedReader = new BufferedReader(new FileReader("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\test_tweets.csv"));
            while((line = bufferedReader.readLine())!= null){
                convertTestDataToArrayList(line);
            }
        }catch(IOException e){
            System.out.println(e.getMessage());
        }
    }

    public static void  readDataSet(){
        BufferedReader bufferedReader = null;

        String line;
        try{
            bufferedReader = new BufferedReader(new FileReader("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\train.csv"));
            while((line = bufferedReader.readLine())!= null){
                convertDataToArrayList(line);
            }
        }catch(IOException e){
           System.out.println(e.getMessage());
        }
    }

    public static void convertDataToArrayList(String line){
        if(line != null){
            String [] splits = line.split(",");
            modelList.add(new Model(splits[0],splits[1],splits[2]));
        }
    }

    public static void convertTestDataToArrayList(String line){
        if(line != null){
            String [] splits = line.split(",");
            modelList.add(new Model(splits[0],splits[1]));
        }
    }

    private static void tokenizeTweets(){
        System.out.print("-------------------------TOKENIZING--------------------"+"\n\n");
        long tokenizerStartTime = System.currentTimeMillis();
        //Tokenizer
        try (InputStream modelIn = new FileInputStream("C:\\Users\\Keshato_Tech_1\\IdeaProjects\\HateSpeechDetector\\src\\main\\resources\\en-token.bin")) {
            TokenizerModel model = new TokenizerModel(modelIn);
            TokenizerME tokenDetector = new TokenizerME(model);
            for(Model modelTweet: modelList){
                if(isTrain){
                    tokenizedModels.add(new Model(modelTweet.getId(),modelTweet.getLabel(),modelTweet.getText(),tokenDetector.tokenize(modelTweet.getText())));
                }
                else {
                    tokenizedModels.add(new Model(modelTweet.getId(),modelTweet.getText(),tokenDetector.tokenize(modelTweet.getText())));
                }
            }
        } catch (IOException e) {
            System.out.print(e.getMessage());
        }

        long tokenizerEndTime = System.currentTimeMillis();
        System.out.println("\n\n Time taken :"+ (tokenizerEndTime - tokenizerStartTime)+"ms \n\n");
        System.out.println(tokenizedModels.size());
    }

    private static void readStopWords() {
        try {
            Scanner scanner = new Scanner(stopWordsFile);
            while (scanner.hasNext()) {
                stopWords.add(scanner.next().toLowerCase());
            }
        } catch (IOException e) {
            System.out.print(e.getMessage());
        }
    }

        private static void removeStopWords() {
            System.out.print("-------------------------REMOVING STOP WORDS--------------------"+"\n\n");

            for(Model model: tokenizedModels){
                for(int i= 0; i<model.getTokens().length;i++) {
                    String currentToken = model.getTokens()[i].toLowerCase();
                    if (!stopWords.contains(currentToken)) {
                        stopWordRemovedTokens.add(currentToken);
                   }
                }
                //create a temp array to hold the values of tokens without stop words using temp list array
                String [] temp = new String[stopWordRemovedTokens.size()];
                //Add the stop word removed tokens to the String array
                for(int d = 0;d<stopWordRemovedTokens.size();d++){
                    temp[d]= stopWordRemovedTokens.get(d);
                }
                //Add a new Document to the stop word removed tokens list
                if(isTrain){
                    stopWordRemoved.add(new Model(model.getId(),model.getLabel(),model.getText(),temp));
                    stopWordRemovedTokens.clear();
                }
                else{
                    stopWordRemoved.add(new Model(model.getId(),model.getText(),temp));
                    stopWordRemovedTokens.clear();
                }

            }
//            System.out.print("STOP WORD REMOVED TOKEN ARRAY SIZE = "+stopWordRemovedTokens.size()+"\n\n");

        }

       private static void filterModels() {
        long tokenFilterStart = System.currentTimeMillis();
        System.out.print("-------------------------FILTERING TOKENS--------------------"+"\n\n");

     //   filter out non-characters and words with less than 3
        for(Model document: stopWordRemoved){
            String [] filteredDocument = Arrays.stream(document.getTokens()).map(s -> s.replaceAll("[^a-zA-Z]",""))
                    .filter(s -> s.length()>2)
                    .toArray(String []::new);
            if(isTrain){
                filteredModels.add(new Model(document.getId(),document.getLabel(),document.getText(),filteredDocument));
            }
            else {
                filteredModels.add(new Model(document.getId(),document.getText(),filteredDocument));
            }

//            for(int i = 0; i<filteredDocument.length-1;i++){
//                System.out.println(filteredDocument[i]);
//            }
        }
        long tokenFilterEnd = System.currentTimeMillis();
        System.out.println("\n\n Time taken :"+ (tokenFilterEnd - tokenFilterStart)+"ms \n\n");
        System.out.print("FILTERED TOKEN ARRAY SIZE = "+filteredModels.size()+"\n\n");
    }

    private static void stemTokens() {
        String[] stems = new String[filteredModels.size()];

        PorterStemmer porterStemmer = new PorterStemmer();
        for(Model document: filteredModels){
            String [] tempDocument = new String[document.getTokens().length];
            for(int i = 0; i<document.getTokens().length;i++){
                tempDocument[i] = porterStemmer.stem(document.getTokens()[i]);
            }
            if(isTrain){
                stemmedDocuments.add(new Model(document.getId(),document.getLabel(),document.getText(),tempDocument));
            }
            else {
                stemmedDocuments.add(new Model(document.getId(),document.getText(),tempDocument));
            }
        }
        System.out.print("-------------------------STEMS--------------------"+"\n\n");
//        for(int i = 0; i< stemmedDocuments.size();i++){
//            System.out.println(stemmedDocuments.get(i).getLabel()+" "+stemmedDocuments.get(i).toString());
//        }

    }

    public static List<Model> getPreprocessedCorpus(){
        return filteredModels;
    }

}

