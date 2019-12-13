package com.jobreadyprogrammer.spark;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KmeansClustering extends SparkSessionAbstract {

    public static void main(String[] args) {
        SparkSession sparkSession = getSparkSession();
        Dataset<Row> wholeSaleDf = getDatasetInferingSchema(sparkSession, "csv", "src/main/resources/Wholesale customers data.csv");
        wholeSaleDf.show();
        Dataset<Row> featuresDf = wholeSaleDf.select("channel", "region", "fresh", "milk", "grocery", "frozen", "detergents_paper", "delicassen");
        VectorAssembler assembler = new VectorAssembler();
        assembler.setInputCols(new String[]{"channel", "region", "fresh", "milk", "grocery", "frozen", "detergents_paper", "delicassen"})
        .setOutputCol("features");
        Dataset<Row> trainingData = assembler.transform(featuresDf).select("features");
        KMeans kMeans = new KMeans();
        kMeans.setK(3);
        KMeansModel model = kMeans.fit(trainingData);
        System.out.println(model.computeCost(trainingData));
        model.summary().predictions().show();
    }
}
