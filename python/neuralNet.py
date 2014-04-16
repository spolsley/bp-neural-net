import sys
import numpy as np

class neuralNet:
	def __init__(self, sizeInput, sizeHidden, sizeOutput):
		self.learningRate = 0.4
		self.momentum = 0.8

		self.sizeInput = sizeInput
		self.sizeHidden = sizeHidden
		self.sizeOutput = sizeOutput

		# Allocate neurons
		self.neuronsInput = np.zeros((1,sizeInput))
		self.neuronsHidden = np.zeros((1,sizeHidden+1))
		self.neuronsOutput = np.zeros((1,sizeOutput+1))

		# Allocate weights
		self.weightsInputToHidden = np.random.random((sizeInput+1,sizeHidden)) * 0.24 - 0.12
		self.weightsHiddenToOutput = np.random.random((sizeHidden+1,sizeOutput)) * 0.24 - 0.12

		# Allocate training data
		self.errorHidden = np.zeros((1,sizeHidden+1))
		self.errorOutput = np.zeros((1,sizeOutput))
		self.idealOutput = np.zeros((1,sizeOutput))
		self.oldDeltaWeightInputToHidden = np.zeros((sizeInput+1,sizeHidden))
		self.oldDeltaWeightHiddenToOutput = np.zeros((sizeHidden+1,sizeOutput))
		self.newDeltaWeightInputToHidden = np.zeros((sizeInput+1,sizeHidden))
		self.newDeltaWeightHiddenToOutput = np.zeros((sizeHidden+1,sizeOutput))

	def activationFunction(self, in_num):
		# Many functions may be used for activation
		# Sigmoid used by default
		return 1.0 / (1.0 + np.exp(-in_num))

	def feedForward(self, in_arr):
		self.neuronsInput = np.concatenate((in_arr,np.array([[-1]])),axis=1)
		self.neuronsHidden = np.concatenate((self.activationFunction(np.dot(self.neuronsInput,self.weightsInputToHidden)),np.array([[-1]])),axis=1)
		self.neuronsOutput = self.activationFunction(np.dot(self.neuronsHidden,self.weightsHiddenToOutput))

	def saveWeights(self, outFile):
		f = open(outFile, 'w')
		f.write("Dimensions:\n")
		f.write("%d %d %d\n" % (self.sizeInput, self.sizeHidden, self.sizeOutput))

		f.write("Weights Input To Hidden:\n")
		f.writelines(' '.join("%.50f" % j for j in i) + '\n' for i in self.weightsInputToHidden)

		f.write("Weights Hidden To Output:\n")
		f.writelines(' '.join("%.50f" % j for j in i) + '\n' for i in self.weightsHiddenToOutput)
		f.close()
		return True

	def loadWeights(self, inFile):
		f = open(inFile, 'r')
		f.readline() # Dimensions Label
		line = f.readline()
		line = [int(i) for i in line.split()]
		if (line[0] == self.sizeInput) and (line[1] == self.sizeHidden) and (line[2] == self.sizeOutput):
			f.readline() # Weights Label
			weightsIToH = []
			for x in xrange(0,self.sizeInput + 1):
				line = f.readline()
				line = [float(i) for i in line.split()]
				weightsIToH.append(line)

			f.readline() # Weights Label
			weightsHToO = []
			for x in xrange(0,self.sizeHidden + 1):
				line = f.readline()
				line = [float(i) for i in line.split()]
				weightsHToO.append(line)
			self.weightsInputToHidden = np.array(weightsIToH)
			self.weightsHiddenToOutput = np.array(weightsHToO)
			f.close()
			return True
		else:
			return False

	def trainBatch(self, inputs, outputs, maxEpochs):
		numTests = np.size(inputs,0)
		training = int(0.6 * numTests)
		generalizing = int(0.8 * numTests)
		epoch = 1

		while epoch <= 20:
			TMSE = 0
			Tacc = 0
			GMSE = 0
			Gacc = 0
			VMSE = 0
			Vacc = 0
			r = 0
			count = 0
			for i in xrange(0,training):
				# Feed forward and test accuracy
				self.feedForward(inputs[[i],:])

				TMSE = TMSE + np.power(1-self.neuronsOutput[0,outputs[i,0]],2)
				r = np.argmax(self.neuronsOutput)
				if r == outputs[i,0]:
					Tacc = Tacc + 1

				# Back propagation and weight updates
				self.idealOutput = np.zeros((1,self.sizeOutput))
				self.idealOutput[0,outputs[i,0]] = 1
				self.errorOutput = self.neuronsOutput*(1-self.neuronsOutput)*(self.idealOutput-self.neuronsOutput)
				self.errorHidden = self.neuronsHidden*(1-self.neuronsHidden)*sum(np.transpose(self.weightsHiddenToOutput*self.errorOutput))
				self.newDeltaWeightHiddenToOutput = self.learningRate*np.dot(np.transpose(self.neuronsHidden),self.errorOutput) + self.momentum*self.oldDeltaWeightHiddenToOutput
				self.weightsHiddenToOutput = self.weightsHiddenToOutput + self.newDeltaWeightHiddenToOutput
				self.newDeltaWeightInputToHidden = self.learningRate*np.dot(np.transpose(self.neuronsInput),self.errorHidden[[0],0:-1]) + self.momentum*self.oldDeltaWeightInputToHidden
				self.weightsInputToHidden = self.weightsInputToHidden + self.newDeltaWeightInputToHidden

				self.oldDeltaWeightHiddenToOutput = self.newDeltaWeightHiddenToOutput
				self.oldDeltaWeightInputToHidden = self.newDeltaWeightInputToHidden

				count = count + 1
			TMSE = TMSE / count
			Tacc = Tacc / float(count) * 100.0

			count = 0
			for i in xrange(training,generalizing):
				self.feedForward(inputs[[i],:])

				GMSE = GMSE + np.power(1-self.neuronsOutput[0,outputs[i,0]],2)
				r = np.argmax(self.neuronsOutput)
				if r == outputs[i,0]:
					Gacc = Gacc + 1

				count = count + 1
			GMSE = GMSE / count
			Gacc = Gacc / float(count) * 100.0

			print("Epoch: %d, TMSE: %f, Tacc: %f, GMSE: %f, Gacc: %f" % (epoch,TMSE,Tacc,GMSE,Gacc))
			sys.stdout.flush()

			if (((TMSE < 0.05) and (GMSE < 0.05)) or ((Tacc > 95) and (Gacc > 95))):
				break
			epoch = epoch + 1

		count = 0
		for i in xrange(generalizing,numTests):
			self.feedForward(inputs[[i],:])

			VMSE = VMSE + np.power(1-self.neuronsOutput[0,outputs[i,0]],2)
			r = np.argmax(self.neuronsOutput)
			if r == outputs[i,0]:
				Vacc = Vacc + 1

			count = count + 1
		VMSE = VMSE / count
		Vacc = Vacc / float(count) * 100.0

		print("VMSE: %f, Vacc: %f" % (VMSE, Vacc))

	def trainLive(self, input, output):
		# Feed forward
		self.feedForward(input)

		# Back propagation and weight updates
		self.idealOutput = np.zeros((1,self.sizeOutput))
		self.idealOutput[0,output] = 1
		self.errorOutput = self.neuronsOutput*(1-self.neuronsOutput)*(self.idealOutput-self.neuronsOutput)
		self.errorHidden = self.neuronsHidden*(1-self.neuronsHidden)*sum(np.transpose(self.weightsHiddenToOutput*self.errorOutput))
		self.newDeltaWeightHiddenToOutput = self.learningRate*np.dot(np.transpose(self.neuronsHidden),self.errorOutput) + self.momentum*self.oldDeltaWeightHiddenToOutput
		self.weightsHiddenToOutput = self.weightsHiddenToOutput + self.newDeltaWeightHiddenToOutput
		self.newDeltaWeightInputToHidden = self.learningRate*np.dot(np.transpose(self.neuronsInput),self.errorHidden[[0],0:-1]) + self.momentum*self.oldDeltaWeightInputToHidden
		self.weightsInputToHidden = self.weightsInputToHidden + self.newDeltaWeightInputToHidden

		self.oldDeltaWeightHiddenToOutput = self.newDeltaWeightHiddenToOutput
		self.oldDeltaWeightInputToHidden = self.newDeltaWeightInputToHidden

		return self.neuronsOutput

	def classify(self,input):
		self.feedForward(input)
		return np.argmax(self.neuronsOutput)
