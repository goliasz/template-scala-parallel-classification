package org.template.classification

import io.prediction.controller.EngineFactory
import io.prediction.controller.Engine

class Query(
  val attr0 : Double,
  val attr1 : Double,
  val attr2 : Double,
  val attr3 : Double,
  val attr4 : Double,
  val attr5 : Double,
  val attr6 : Double,
  val attr7 : Double,
  val attr8 : Double,
  val attr9 : Double,
  val attr10 : Double,
  val attr11 : Double,
  val attr12 : Double,
  val attr13 : Double,
  val attr14 : Double,
  val attr15 : Double,
  val attr16 : Double,
  val attr17 : Double,
  val attr18 : Double,
  val attr19 : Double,
  val attr20 : Double,
  val attr21 : Double,
  val attr22 : Double,
  val attr23 : Double,
  val attr24 : Double
) extends Serializable

class PredictedResult(
  val label: Double
) extends Serializable

class ActualResult(
  val label: Double
) extends Serializable

object ClassificationEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("naive" -> classOf[NaiveBayesAlgorithm]),
      classOf[Serving])
  }
}
