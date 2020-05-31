from pyspark.mllib.random import RandomRDDs
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import coalesce  

sc= SparkContext()
spark=SparkSession(sc)

Rdd=RandomRDDs.uniformRDD(sc,100044,2) 
Rdd2=RandomRDDs.uniformRDD(sc,100044,2)
Rdd_cons=Rdd.map(lambda x : 102.83547008547009+ 102.85047727*x)
Rdd_cons=Rdd_cons.sortBy(lambda x: x)
Rdd_pop=Rdd2.map(lambda x : 3401 + 150000*x)
Rdd_pop=Rdd_pop.sortBy(lambda x: x)
Rdd_pop=Rdd_pop.map(lambda x: int(x+6071639))
mois=[]
for i in range(100044):
	mois.append(i+1)
Rdd_mois=sc.parallelize(mois,2)
colone1=Row("consomation")
colone2=Row("population")
colone3=Row("mois")
df_cons=Rdd_cons.map(colone1).toDF()
df_pop=Rdd_pop.map(colone2).toDF()
df_mois=Rdd_mois.map(colone3).toDF()
df_mois=df_mois.withColumn('ligne_id', f.monotonically_increasing_id())
df_pop=df_pop.withColumn('ligne_id', f.monotonically_increasing_id())
df_cons=df_cons.withColumn('ligne_id', f.monotonically_increasing_id())
df=df_mois.join(df_pop, on=["ligne_id"]).sort("ligne_id")
df=df.join(df_cons, on=["ligne_id"]).sort("ligne_id")
df=df.drop("ligne_id")
df.coalesce(1).write.csv('dataset.csv')




