import java.io._

import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions._

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

import org.apache.spark.mllib.linalg.Vectors

val h = scala.io.Source.fromFile("kddcup.names").mkString.split("\n").drop(1)
var structs = Array[StructField]()
for (i <- 0 until h.length) {
   var cur = h(i).replace(".","").split(": ")
   if (cur(1) == "continuous")
      structs :+= StructField(cur(0),DoubleType)
   else
      structs :+= StructField(cur(0),StringType) 
}
structs :+= StructField("outcome",StringType)
val custom = StructType(structs)


var df = spark.read.option("header", false).schema(custom).csv("hdfs://localhost:9000/data/kddcup.data.corrected")

val removeP = udf((x: String) => x.replace(".",""))
val makeBinary = udf((x: String) => if (x=="normal") 1 else 0)
df = df.withColumn("outcome",removeP(col("outcome")))
df = df.withColumn("binary_outcome",makeBinary(col("outcome")))
df = df.withColumn("id",monotonically_increasing_id())

val indexer = new StringIndexer()
indexer.setInputCol("protocol_type")
indexer.setOutputCol("protocol_type_index")
indexer.setHandleInvalid("error")
val protocol_type = indexer.fit(df)

val indexer = new StringIndexer()
indexer.setInputCol("logged_in")
indexer.setOutputCol("logged_in_index")
indexer.setHandleInvalid("error")
val logged_in = indexer.fit(df)

val indexer = new StringIndexer()
indexer.setInputCol("is_guest_login")
indexer.setOutputCol("is_guest_login_index")
indexer.setHandleInvalid("error")
val is_guest_login = indexer.fit(df)

val indexer = new StringIndexer()
indexer.setInputCol("service")
indexer.setOutputCol("service_index")
indexer.setHandleInvalid("error")
val service = indexer.fit(df)

val indexer = new StringIndexer()
indexer.setInputCol("flag")
indexer.setOutputCol("flag_index")
indexer.setHandleInvalid("error")
val flag = indexer.fit(df)

df = protocol_type.transform(df).drop("protocol_type")
df = logged_in.transform(df).drop("logged_in")
df = service.transform(df).drop("service")
df = flag.transform(df).drop("flag")
df = is_guest_login.transform(df).drop("is_guest_login")

for (i <- List("is_host_login","num_outbound_cmds","land")) {
   df = df.drop(i)
}

val exclude = List("outcome","binary_outcome","id")
val model_vars = (df.columns.toSet -- exclude.toSet).toList
val model_vars_row = "binary_outcome" :: model_vars

val categoricals = for (i<-0 until model_vars.length if model_vars(i).contains("index")) yield (i,model_vars(i))
var tempMap = scala.collection.mutable.Map[Int,Int]()
for (i <- categoricals) {
   tempMap(i._1) = i._2 match {
      case "protocol_type_index" => protocol_type.labels.length
      case "logged_in_index" => logged_in.labels.length
      case "service_index" => service.labels.length
      case "flag_index" => flag.labels.length
      case "is_guest_login_index" => is_guest_login.labels.length
   }
}
val catmap = tempMap.toMap

val crossfolds = df.randomSplit(weights=Array.fill(10)(0.1),seed=1028)

var auc = Array[Double]()

for (i <- 0 until crossfolds.length) {
   val trainSet = ((0 to 9).toSet -- Set(i)).toList
   val train = (for (x <- trainSet) yield crossfolds(x).select(model_vars_row.head,model_vars_row.tail: _*).rdd.map(y => LabeledPoint(y.getInt(0),Vectors.dense((for (z <- 1 until y.length) yield y.getDouble(z)).toArray)))).reduceLeft {_.union(_)}
   val test = crossfolds(i).select(model_vars_row.head,model_vars_row.tail: _*).rdd.map(y => LabeledPoint(y.getInt(0),Vectors.dense((for (z <- 1 until y.length) yield y.getDouble(z)).toArray)))
   val testLabel = crossfolds(i).select("binary_outcome").rdd.map(y => y.getInt(0))
   
   val boostingStrategy = BoostingStrategy.defaultParams("Classification")
   boostingStrategy.learningRate = 0.1
   boostingStrategy.numIterations = 50
   boostingStrategy.treeStrategy.numClasses = 2 
   boostingStrategy.treeStrategy.maxDepth = 1
   boostingStrategy.treeStrategy.maxBins = 100
   boostingStrategy.treeStrategy.categoricalFeaturesInfo = catmap

   val model = GradientBoostedTrees.train(train, boostingStrategy)

   // Evaluate model on test instances and compute test error
   val predAndLabel = test.map {point =>
      val prediction = model.predict(point.features)
      (prediction,point.label)
   }
   val metrics = new BinaryClassificationMetrics(predAndLabel)

   auc :+= metrics.areaUnderROC
   println("Fold " + i + " AUC: " + auc.last)
}

val out = new PrintWriter(new File("auc_scala.txt"))
out.write("gbt" + "," + auc.map(_.toString).reduce {_+","+_})
out.close()

