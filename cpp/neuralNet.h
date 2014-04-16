#ifndef NeuralNet
#define NeuralNet

class neuralNet
{
	private:
		// Layer Sizes - Set at initialization
		int sizeInput, sizeHidden, sizeOutput;

		// Layer Neurons - Dynamically allocated arrays
		double* neuronsInput;
		double* neuronsHidden;
		double* neuronsOutput;

		// Layer Weights - Dynamically allocated arrays
		double** weightsInputToHidden;
		double** weightsHiddenToOutput;

		// Training Data - Private data used for training
		double learningRate;
		double momentum;

		double* errorHidden;
		double* errorOutput;
		double* idealOutput;
		double** oldDeltaWeightInputToHidden;
		double** oldDeltaWeightHiddenToOutput;
		double** newDeltaWeightInputToHidden;
		double** newDeltaWeightHiddenToOutput;

		// Functions - Private member functions
		void initializeWeights();
		double activationFunction(double in);
		void feedForward(double* in);

	public:
		// Constructor and Destructor
		neuralNet(int numIn, int numHid, int numOut);
		~neuralNet();

		// Saving and Loading
		bool saveWeights(char* outFile);
		bool loadWeights(char* inFile);

		// Training and Classifying
		void setLearningRate(double lr);
		void setMomentum(double m);
		double getLearningRate();
		double getMomentum();

		void trainBatch(double** inputs, int* outputs, int numTests, int maxEpochs);
		double* trainLive(double* in, int out);
		int classify(double* in);
};

#include "neuralNet.cpp"
#endif
