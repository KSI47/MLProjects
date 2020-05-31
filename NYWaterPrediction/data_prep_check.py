from pyspark.mllib.regression import LinearRegressionWithSGD, LinearRegressionModel, LabeledPoint
from pyspark import SparkContext

def parsePoint(line):

	
	values=[float(x) for x in line.split(',')]
	
		
	return LabeledPoint(values[2],[values[0],values[1]])

sc=SparkContext()

data=sc.textFile("/home/khaled/project/data_gen/water_data.csv")

parseData=data.map(parsePoint)

itr_data=parseData.collect()

f=open("data","w+")

f.write("training data \n")

for l in itr_data: 
	f.write(str(l)+"\n")


data_test=sc.textFile("/home/khaled/project/data_gen/test.csv")

parseData_test=data_test.map(parsePoint)

itr_data_test=parseData_test.collect()
f.write("test data \n")
for l in itr_data_test: 
	f.write(str(l)+"\n")
f.close()


