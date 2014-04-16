import sys
import numpy as np
from neuralNet import neuralNet

with open('input.txt') as f:
	inputs = []
	for line in f:
		line = line.split()
		if line:
			line = [float(i) for i in line]
			inputs.append(line)

with open('output.txt') as f:
	outputs = []
	for line in f:
		line = line.split()
		if line:
			line = [int(i) for i in line]
			outputs.append(line)

input = np.array(inputs)
output = np.array(outputs)

nn = neuralNet(400,30,10)

# Training
# ---

# Batch training
nn.trainBatch(input,output,20)

# Live training
#tests = np.size(input,0)
#acc = 0
#for i in xrange(0, tests):
#	if (np.argmax(nn.trainLive(input[[i],:],output[i,0])) == output[i,0]):
#		acc = acc + 1
#acc = acc / float(tests) * 100
#print("Live training accuracy: %f" % (acc))

# Save/Load
# ---

# Saving weights
#nn.saveWeights('saved.txt')

# Loading weights
#nn.loadWeights('saved.txt')

print("Value: %d, Result: %d" % (output[20,0],nn.classify(input[[20],:])))
print("Value: %d, Result: %d" % (output[300,0],nn.classify(input[[300],:])))
print("Value: %d, Result: %d" % (output[2500,0],nn.classify(input[[2500],:])))
print("Value: %d, Result: %d" % (output[4800,0],nn.classify(input[[4800],:])))
