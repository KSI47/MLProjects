from pyspark.mllib.regression import LinearRegressionWithSGD, LinearRegressionModel, LabeledPoint
from pyspark import SparkContext

def parsePoint(line):

	values=[float(x) for x in line.split(',')]
	
	return LabeledPoint(values[2],[values[0], values[1]])# consommation,annee,population

sc=SparkContext()

data=sc.textFile("/home/khaled/project/data_gen/data_train.csv")

parseData=data.map(parsePoint)

model = LinearRegressionWithSGD.train(parseData, iterations=10022, step=0.0000000000000276)

valuesAndPreds = parseData.map(lambda p: (p.label, model.predict(p.features)))

MSE = valuesAndPreds\
	.map(lambda vp: (vp[0]-vp[1])**2)\
	.reduce(lambda x,y: x+y)/valuesAndPreds.count()
f=open("mse_rlst.txt","w+")
f.write(" erreur quadratique moyenne = " + str(MSE))
f.close()

model.save(sc, "/home/khaled/project/tmp/lin_reg_model")





