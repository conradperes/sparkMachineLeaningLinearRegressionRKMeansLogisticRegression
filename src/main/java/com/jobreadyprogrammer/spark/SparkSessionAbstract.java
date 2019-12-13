package com.jobreadyprogrammer.spark;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class SparkSessionAbstract {
    protected static Dataset<Row> getDatasetInferingSchema(SparkSession spark, String fileType, String fileName) {
        return spark.read().format(fileType)
                .option("inferSchema", "true")
                .option("header", true)
                .load(fileName);
    }


    protected static SparkSession getSparkSession() {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        System.setProperty("hadoop.home.dir","C:/hadoop" );
        return SparkSession.builder()
                .appName("Leaning SparkSession SQL Dataframe API")
                .master("local")
                .getOrCreate();
    }
}
