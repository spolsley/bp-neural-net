#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cstdlib>

#include "neuralNet.h"

using namespace std;

// Constructor
neuralNet::neuralNet(int numIn, int numHid, int numOut) : sizeInput(numIn), sizeHidden(numHid), sizeOutput(numOut)
{
	learningRate = 0.4;
	momentum = 0.8;

	// Allocate neurons
	neuronsInput = new double[sizeInput + 1];
	neuronsHidden = new double[sizeHidden + 1];
	neuronsOutput = new double[sizeOutput];

	// Allocate weights
	weightsInputToHidden = new double*[sizeInput + 1];
	weightsHiddenToOutput = new double*[sizeHidden + 1];

	// Allocate training data
	errorHidden = new double[sizeHidden + 1];
	errorOutput = new double[sizeOutput];
	idealOutput = new double[sizeOutput];
	oldDeltaWeightInputToHidden = new double*[sizeInput + 1];
	oldDeltaWeightHiddenToOutput = new double*[sizeHidden + 1];
	newDeltaWeightInputToHidden = new double*[sizeInput + 1];
	newDeltaWeightHiddenToOutput = new double*[sizeHidden + 1];

	// Initialize layers
	for (int i = 0; i <= sizeInput; i++)
	{
		neuronsInput[i] = 0;
		weightsInputToHidden[i] = new double[sizeHidden];
		oldDeltaWeightInputToHidden[i] = new double[sizeHidden];
		newDeltaWeightInputToHidden[i] = new double[sizeHidden];
		for (int j = 0; j < sizeHidden; j++)
		{
			weightsInputToHidden[i][j] = 0;
			oldDeltaWeightInputToHidden[i][j] = 0;
			newDeltaWeightInputToHidden[i][j] = 0;
		}
	}
	neuronsInput[sizeInput] = -1; // Bias neuron
	for (int j = 0; j <= sizeHidden; j++)
	{
		neuronsHidden[j] = 0;
		errorHidden[j] = 0;
		weightsHiddenToOutput[j] = new double[sizeOutput];
		oldDeltaWeightHiddenToOutput[j] = new double[sizeOutput];
		newDeltaWeightHiddenToOutput[j] = new double[sizeOutput];
		for (int k = 0; k < sizeOutput; k++)
		{
			weightsHiddenToOutput[j][k] = 0;
			oldDeltaWeightHiddenToOutput[j][k] = 0;
			newDeltaWeightHiddenToOutput[j][k] = 0;
		}
	}
	neuronsHidden[sizeHidden] = -1; // Bias neuron
	for (int k = 0; k < sizeOutput; k++)
	{
		neuronsOutput[k] = 0;
		errorOutput[k] = 0;
		idealOutput[k] = 0;
	}

	initializeWeights();
}

// Destructor
neuralNet::~neuralNet()
{
	for (int i = 0; i <= sizeInput; i++)
	{
		delete[] weightsInputToHidden[i];
		delete[] oldDeltaWeightInputToHidden[i];
		delete[] newDeltaWeightInputToHidden[i];
	}
	for (int j = 0; j <= sizeHidden; j++)
	{
		delete[] weightsHiddenToOutput[j];
		delete[] oldDeltaWeightHiddenToOutput[j];
		delete[] newDeltaWeightHiddenToOutput[j];
	}
	delete[] weightsInputToHidden;
	delete[] oldDeltaWeightInputToHidden;
	delete[] newDeltaWeightInputToHidden;
	delete[] weightsHiddenToOutput;
	delete[] oldDeltaWeightHiddenToOutput;
	delete[] newDeltaWeightHiddenToOutput;

	delete[] errorHidden;
	delete[] errorOutput;
	delete[] idealOutput;

	delete[] neuronsInput;
	delete[] neuronsHidden;
	delete[] neuronsOutput;
}

void neuralNet::initializeWeights()
{
	// Range of weights
	double rangeHidden = 1/sqrt((double)sizeInput);
	double rangeOutput = 1/sqrt((double)sizeHidden);

	// Randomly set weights
	for (int i = 0; i <= sizeInput; i++)
	{
		for (int j = 0; j < sizeHidden; j++)
			weightsInputToHidden[i][j] = (((double)(rand() % 100 + 1))/100.0) * 2.0 * rangeHidden - rangeHidden;
	}
	for (int j = 0; j <= sizeHidden; j++)
	{
		for (int k = 0; k < sizeOutput; k++)
			weightsHiddenToOutput[j][k] = (((double)(rand() % 100 + 1))/100.0) * 2.0 * rangeOutput - rangeOutput;
	}
}

double neuralNet::activationFunction(double in)
{
	// Many functions may be used for activation
	// Sigmoid used by default
	return 1.0 / (1.0 + exp(-in));
}

void neuralNet::feedForward(double* in)
{
	for (int i = 0; i < sizeInput; i++)
		neuronsInput[i] = in[i];
	for (int j = 0; j < sizeHidden; j++)
	{
		neuronsHidden[j] = 0;
		for (int i = 0; i <= sizeInput; i++)
			neuronsHidden[j] += neuronsInput[i] * weightsInputToHidden[i][j];
		neuronsHidden[j] = activationFunction(neuronsHidden[j]);
	}
	for (int k = 0; k < sizeOutput; k++)
	{
		neuronsOutput[k] = 0;
		for (int j = 0; j <= sizeHidden; j++)
			neuronsOutput[k] += neuronsHidden[j] * weightsHiddenToOutput[j][k];
		neuronsOutput[k] = activationFunction(neuronsOutput[k]);
	}
}

void neuralNet::setLearningRate(double lr)
{
	learningRate = lr;
}

void neuralNet::setMomentum(double m)
{
	momentum = m;
}

double neuralNet::getLearningRate()
{
	return learningRate;
}

double neuralNet::getMomentum()
{
	return momentum;
}

bool neuralNet::saveWeights(char* outFile)
{
	fstream output;
	output.open(outFile, ios::out);

	if (output.is_open())
	{
		output.precision(50);

		output << "Dimensions:\n" << sizeInput << " " << sizeHidden << " " << sizeOutput << endl;

		output << "Weights Input To Hidden:\n";
		for (int i = 0; i <= sizeInput; i++)
		{
			for (int j = 0; j < sizeHidden; j++)
				output << weightsInputToHidden[i][j] << " ";
			output << endl;
		}

		output << "Weights Hidden To Output:\n";
		for (int j = 0; j <= sizeHidden; j++)
		{
			for (int k = 0; k < sizeOutput; k++)
				output << weightsHiddenToOutput[j][k] << " ";
			output << endl;
		}

		output.close();
		return true;
	}
	else
	{
		return false;
	}
}

bool neuralNet::loadWeights(char* inFile)
{
	fstream input;
	input.open(inFile, ios::in);

	if (input.is_open())
	{
		int numInput, numHidden, numOutput;
		string line = "";

		getline(input, line); // Dimensions Label
		input >> numInput >> numHidden >> numOutput;
		getline(input, line); // Clear line feed and newline characters

		if ((numInput == sizeInput) && (numHidden == sizeHidden) && (numOutput == sizeOutput))
		{
			getline(input, line); // Weights Label
			for (int i = 0; i <= sizeInput; i++)
			{
				for (int j = 0; j < sizeHidden; j++)
					input >> weightsInputToHidden[i][j];
				getline(input, line); // Clear line feed and newline characters
			}

			getline(input, line); // Weights Label
			for (int j = 0; j <= sizeHidden; j++)
			{
				for (int k = 0; k < sizeOutput; k++)
					input >> weightsHiddenToOutput[j][k];
				getline(input, line); // Clear line feed and newline characters
			}
			input.close();
			return true;
		}
		else
		{
			input.close();
			return false;
		}
	}
	else
	{
		return false;
	}
}

void neuralNet::trainBatch(double** inputs, int* outputs, int numTests, int maxEpochs)
{
	int training = 0.6 * numTests;
	int generalizing = 0.8 * numTests;

	double TMSE, Tacc, GMSE, Gacc, VMSE, Vacc;
	double max, sum;
	int result, count;

	for (int epoch = 1; epoch <= maxEpochs; epoch++)
	{
		TMSE = 0;
		Tacc = 0;
		GMSE = 0;
		Gacc = 0;
		// Train on 60% of the data
		count = 0;
		for (int test = 0; test < training; test++)
		{
			max = -1.0;
			result = 0;

			// Feed forward and test accuracy
			feedForward(inputs[test]);

			TMSE += pow((1.0 - neuronsOutput[outputs[test]]),2.0);
			for (int k = 0; k < sizeOutput; k++)
			{
				idealOutput[k] = 0;
				if (neuronsOutput[k] > max)
				{
					max = neuronsOutput[k];
					result = k;
				}
			}
			if (result == outputs[test])
				Tacc += 1.0;

			// Back propagation
			idealOutput[outputs[test]] = 1;
			for (int k = 0; k < sizeOutput; k++)
			{
				errorOutput[k] = neuronsOutput[k]*(1-neuronsOutput[k])*(idealOutput[k]-neuronsOutput[k]);
			}
			for (int j = 0; j <= sizeHidden; j++)
			{
				sum = 0;
				for (int k = 0; k < sizeOutput; k++)
				{
					sum += weightsHiddenToOutput[j][k]*errorOutput[k];
					newDeltaWeightHiddenToOutput[j][k] = (learningRate*neuronsHidden[j]*errorOutput[k]) + momentum*oldDeltaWeightHiddenToOutput[j][k];
					oldDeltaWeightHiddenToOutput[j][k] = newDeltaWeightHiddenToOutput[j][k];
					weightsHiddenToOutput[j][k] += newDeltaWeightHiddenToOutput[j][k];
				}
				errorHidden[j] = neuronsHidden[j]*(1-neuronsHidden[j])*sum;
			}
			for (int i = 0; i <= sizeInput; i++)
			{
				for (int j = 0; j < sizeHidden; j++)
				{
					newDeltaWeightInputToHidden[i][j] = (learningRate*neuronsInput[i]*errorHidden[j]) + momentum*oldDeltaWeightInputToHidden[i][j];
					oldDeltaWeightInputToHidden[i][j] = newDeltaWeightInputToHidden[i][j];
					weightsInputToHidden[i][j] += newDeltaWeightInputToHidden[i][j];
				}
			}
			count++;
		}
		TMSE = TMSE / (double)count;
		Tacc = Tacc / (double)count * 100.0;

		// Generalization testing on 20% of the data
		count = 0;
		for (int test = training; test < generalizing; test++)
		{
			max = -1.0;
			result = 0;

			feedForward(inputs[test]);

			GMSE += pow((1 - neuronsOutput[outputs[test]]),2);
			for (int k = 0; k < sizeOutput; k++)
			{
				if (neuronsOutput[k] > max)
				{
					max = neuronsOutput[k];
					result = k;
				}
			}
			if (result == outputs[test])
				Gacc += 1.0;
			count++;
		}
		GMSE = GMSE / (double)count;
		Gacc = Gacc / (double)count * 100.0;

		cout<<"Epoch: "<<epoch<<", TMSE: "<<TMSE<<", Tacc: "<<Tacc<<", GMSE: "<<GMSE<<", Gacc: "<<Gacc<<endl;
		if (((TMSE < 0.05)&&(GMSE < 0.05)) || ((Tacc > 95)&&(Gacc > 95)))
			break;
	}

	// Validating results
	count = 0;
	for (int test = generalizing; test < numTests; test++)
	{
		max = -1.0;
		result = 0;

		feedForward(inputs[test]);

		VMSE += pow((1 - neuronsOutput[outputs[test]]),2);
		for (int k = 0; k < sizeOutput; k++)
		{
			if (neuronsOutput[k] > max)
			{
				max = neuronsOutput[k];
				result = k;
			}
		}
		if (result == outputs[test])
			Vacc += 1.0;
		count++;
	}
	VMSE = VMSE / (double)count;
	Vacc = Vacc / (double)count * 100.0;

	cout << "VMSE: " << VMSE << ", Vacc: " << Vacc << endl;
	return;
}

double* neuralNet::trainLive(double* in, int out)
{
	double sum;

	// Feed forward
	feedForward(in);

	for (int k = 0; k < sizeOutput; k++)
		idealOutput[k] = 0;

	// Back propagation
	idealOutput[out] = 1;
	for (int k = 0; k < sizeOutput; k++)
	{
		errorOutput[k] = neuronsOutput[k]*(1-neuronsOutput[k])*(idealOutput[k]-neuronsOutput[k]);
	}
	for (int j = 0; j <= sizeHidden; j++)
	{
		sum = 0;
		for (int k = 0; k < sizeOutput; k++)
		{
			sum += weightsHiddenToOutput[j][k]*errorOutput[k];
			newDeltaWeightHiddenToOutput[j][k] = (learningRate*neuronsHidden[j]*errorOutput[k]) + momentum*oldDeltaWeightHiddenToOutput[j][k];
			oldDeltaWeightHiddenToOutput[j][k] = newDeltaWeightHiddenToOutput[j][k];
			weightsHiddenToOutput[j][k] += newDeltaWeightHiddenToOutput[j][k];
		}
		errorHidden[j] = neuronsHidden[j]*(1-neuronsHidden[j])*sum;
	}
	for (int i = 0; i <= sizeInput; i++)
	{
		for (int j = 0; j < sizeHidden; j++)
		{
			newDeltaWeightInputToHidden[i][j] = (learningRate*neuronsInput[i]*errorHidden[j]) + momentum*oldDeltaWeightInputToHidden[i][j];
			oldDeltaWeightInputToHidden[i][j] = newDeltaWeightInputToHidden[i][j];
			weightsInputToHidden[i][j] += newDeltaWeightInputToHidden[i][j];
		}
	}
	return neuronsOutput;
}

int neuralNet::classify(double* in)
{
	double max = -1.0;
	int result = 0;

	feedForward(in);

	for (int k = 0; k < sizeOutput; k++)
	{
		if (neuronsOutput[k] > max)
		{
			max = neuronsOutput[k];
			result = k;
		}
	}

	return result;
}
