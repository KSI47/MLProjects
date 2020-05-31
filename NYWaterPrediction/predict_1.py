from pyspark.mllib.regression import LinearRegressionWithSGD, LinearRegressionModel, LabeledPoint
from pyspark import SparkContext

def parsePoint(line):

	values=[float(x) for x in line.split(',')]
	
	return LabeledPoint(values[2],[values[0], values[1]])


sc=SparkContext()

model = LinearRegressionModel.load(sc, "/home/khaled/project/tmp/lin_reg_model")

data_test=sc.textFile("/home/khaled/project/data_gen/data_test.csv")

data_test_parsed=data_test.map(parsePoint)

#predics=data_test_parsed.map(lambda x :model.predict(x.features))
result=data_test_parsed.map(lambda x :[x,model.predict(x.features),model.predict(x.features)*100/x.label])

data_itr=result.collect()

f=open("predictions.txt","w+")
f.write("sbah el khir \n")
for i in data_itr:
	f.write("(consommation mesurée, [Jour,Population]), consommation estimé, precision):" +str(i) + "%" "\n")
f.close()
