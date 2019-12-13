package com.jobreadyprogrammer.spark;


import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class LinearMarketingVsSales extends SparkSessionAbstract{

    public static void main(String[] args) {
        SparkSession sparkSession = getSparkSession();
        Dataset<Row> markVsSalesDf = getDatasetInferingSchema(sparkSession, "csv", "src/main/resources/marketing-vs-sales.csv");
        Dataset<Row> mldf = markVsSalesDf.withColumnRenamed("sales", "label")
                .select("label", "marketing_spend", "bad_day");

        String[] featureColumns = {"marketing_spend", "bad_day"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");
        Dataset<Row> lblFeaturesDf = assembler.transform(mldf)
                .select("label", "features");
        lblFeaturesDf.na().drop();
        lblFeaturesDf.show();

        //next we need to create a Linear regression model object
        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel learningModel = linearRegression.fit(lblFeaturesDf);
        learningModel.summary().predictions().show();
        System.out.println("R Squared: "+learningModel.summary().r2());// Good statics to go a fine model of prediction variação de 89% em vendas!
        //learningModel.summary().predictions().show();//Mostrar predições de vendas baseado em quanto é investido em marketing
    }

}
