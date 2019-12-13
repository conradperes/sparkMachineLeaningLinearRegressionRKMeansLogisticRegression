package com.jobreadyprogrammer.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LogisticRegressionExample extends SparkSessionAbstract{

    public static void main(String[] args) {
        SparkSession sparkSession = getSparkSession();
        Dataset<Row> treatmentDf = getDatasetInferingSchema(sparkSession, "csv", "src/main/resources/cryotherapy.csv");
        treatmentDf.show();

        Dataset<Row> lblFeatureDf = treatmentDf.withColumnRenamed("Result_of_Treatment", "label")
                .select("label", "sex", "age", "Time", "Number_of_Warts", "Type", "Area");
        lblFeatureDf = lblFeatureDf.na().drop();
        lblFeatureDf.show();

        StringIndexer genderIndexer = new StringIndexer()
                .setInputCol("sex").setOutputCol("sexIndex");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String []{"sexIndex", "age", "Time", "Number_of_Warts", "Type", "Area"})
                .setOutputCol("features");

        Dataset<Row>[] splitData = lblFeatureDf.randomSplit(new double[]{.7, .3});
        Dataset<Row> trainingDf = splitData[0];
        Dataset<Row> testingDf = splitData[1];
        LogisticRegression logisticRegression = new LogisticRegression();
        Pipeline pl = new Pipeline();
        pl.setStages(new PipelineStage[]{genderIndexer, assembler,logisticRegression});

        PipelineModel model = pl.fit(trainingDf);
        Dataset<Row> results = model.transform(testingDf);
        results.show();

    }
}
