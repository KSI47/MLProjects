from pyspark.mllib.regression import LinearRegressionWithSGD, LinearRegressionModel, LabeledPoint
from pyspark import SparkContext

def parsePoint(line):

	values=[float(x) for x in line.split(',')]
	
	return LabeledPoint(values[2],[values[0], values[1]])


sc=SparkContext()

model = LinearRegressionModel.load(sc, "/home/khaled/project/tmp/lin_reg_model")

data_test=sc.textFile("/home/khaled/project/data_gen/test.csv")
data_test_parsed=data_test.map(parsePoint)
data_test2=sc.textFile("/home/khaled/project/data_gen/test2.csv")
data_test_parsed2=data_test2.map(lambda x: x.split(','))
predics=data_test_parsed.map(lambda x :model.predict(x.features))
predics2=data_test_parsed2.map(lambda x :model.predict(x))
data_itr=predics.collect()
data_itr2=predics2.collect()
f=open("predictions.txt","w+")
f.write("sbah el khir \n")
for i in data_itr:
	f.write("the ouput consumption for is:" +str(i) + "\n")
for i in data_itr2:
	f.write("the ouput consumption for is:" +str(i) + "\n")
f.close()
