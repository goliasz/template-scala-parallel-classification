package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class AlgorithmParams(
  lambda: Double
) extends Params

// extends P2LAlgorithm because the MLlib's NaiveBayesModel doesn't contain RDD.
class NaiveBayesAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, NaiveBayesModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): NaiveBayesModel = {
    // MLLib NaiveBayes cannot handle empty training data.
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")

    NaiveBayes.train(data.labeledPoints, ap.lambda)
  }

  def predict(model: NaiveBayesModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(
      Array(query.attr0, query.attr1, query.attr2, query.attr3, query.attr4, query.attr5, query.attr6, query.attr7, query.attr8, query.attr9, 
      query.attr10, query.attr11, query.attr12, query.attr13, query.attr14, query.attr15, query.attr16, query.attr17, query.attr18, query.attr19, 
      query.attr20, query.attr21, query.attr22, query.attr23, query.attr24)
    ))
    new PredictedResult(label)
  }

}
