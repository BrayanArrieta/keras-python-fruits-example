from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# dataset
dataset = numpy.loadtxt("training.csv", delimiter=",",skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[:,0:900]
Y = dataset[:,900:]

# create model
model = Sequential()
model.add(Dense(52, input_dim=900, activation='relu'))
model.add(Dense(900, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# charge dataset for evaluate and test
datasetE = numpy.loadtxt("evaluate.csv", delimiter=",",skiprows=1)
# split into input (X) and output (Y) variables
XE = datasetE[:,0:900]

# calculate predictions
predictions = model.predict(XE)
print(predictions)
# round predictions
rounded = [[round(x[0]),round(x[1]),round(x[2])] for x in predictions[0:3]]
print(rounded)