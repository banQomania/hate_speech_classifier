package main.java;

public class Model {

    private String id;
    private String label;
    private String text;
    private String [] tokens;

    public Model(String id, String label, String text){
        this.id = id;
        this.label = label;
        this.text = text;
    }

    public Model(String id, String text, String[] tokens) {
        this.id = id;
        this.text = text;
        this.tokens = tokens;
    }

    public Model(String id, String text) {
        this.id = id;
        this.text = text;
    }

    public Model(String id, String label, String text, String[] tokens) {
        this.id = id;
        this.label = label;
        this.text = text;
        this.tokens = tokens;
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

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public String[] getTokens() {

        return tokens;
    }

    public void setTokens(String[] tokens) {
        this.tokens = tokens;
    }

    @Override
    public String toString() {
        String arrayString  = "";
        for(String token: tokens){
            if(tokens[tokens.length-1]!= token){
                arrayString += token +" ";
            }
            else{
                arrayString+=token+"";
            }
        }
        return arrayString;
    }
}
