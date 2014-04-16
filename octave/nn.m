clear all;

% Load and shuffle test data
% X holds inputs and y holds expected results
load('sampledata.mat');
Ordered = [y X];
Shuffled = Ordered(randperm(size(Ordered,1)),:);
y = Shuffled(:,1);
X = Shuffled(:,2:end);
clear Ordered;
clear Shuffled;

% Input array is 1xNumInputs; tack -1 onto the end for bias term
% Ex: 4 inputs: 0.0, 0.3, 0.5, 0.1 represented as [0.0, 0.3, 0.5, 0.1, -1]

SizeInput = 400;
SizeHidden = 30;
SizeOutput = 10;
alpha = 0.4; % Learning rate
beta = 0.8; % Momentum

% Randomly initialize the weights between input and hidden and between hidden and output
% Weights read down a column (bias last term) and a single neuron for each column
% Number of rows equals number of weights equals number of inputs plus bias
WeightHidden = rand(SizeInput + 1, SizeHidden) * 0.24 - 0.12;
WeightOutput = rand(SizeHidden + 1, SizeOutput) * 0.24 - 0.12;
DeltaWHOld = zeros(SizeInput + 1, SizeHidden);
DeltaWOOld = zeros(SizeHidden + 1, SizeOutput);
DeltaWHNew = DeltaWHOld;
DeltaWONew = DeltaWOOld;
Input = zeros(1, SizeInput + 1);
Hidden = zeros(1, SizeHidden + 1);
Output = zeros(1, SizeOutput);

sig = @(v) 1.0 ./ (1.0 + exp(-v));
feed = @(i,wh,wo) sig(sig([[X(i,:), -1] * wh, -1]) * wo);
warning("off","Octave:broadcast");

total = size(X,1);
training = 1:total*0.6;
generalization = total*0.6:total*0.8;
validation = total*0.8:total;
epoch = 1;

while (epoch <= 20)
	TMSE = 0;
	Tacc = 0;
	GMSE = 0;
	Gacc = 0;
	VMSE = 0;
	Vacc = 0;
	v = 0;
	r = 0;
	printf('Epoch: %d, ',epoch);
	for i = training
		% Feed forward
		Input = [X(i,:), -1];
		Hidden = sig([Input * WeightHidden, -1]);
		Output = sig(Hidden * WeightOutput);

		v = Output(y(i));
		TMSE = TMSE + (1-v)^2;
		[v,r] = max(Output);
		if y(i) == r
			Tacc++;
		end

		% Back-propagation and weight updates
		Desired = zeros(1,SizeOutput);
		Desired(y(i)) = 1;
		ErrorOutput = Output.*(1-Output).*(Desired-Output);
		ErrorHidden = Hidden.*(1-Hidden).*(sum((WeightOutput.*ErrorOutput)'));
		DeltaWONew = alpha*(Hidden' * ErrorOutput) + beta*DeltaWOOld;
		WeightOutput = WeightOutput + DeltaWONew;
		DeltaWHNew = alpha*(Input' * ErrorHidden(1,1:SizeHidden)) + beta*DeltaWHOld;
		WeightHidden = WeightHidden + DeltaWHNew;

		DeltaWOOld = DeltaWONew;
		DeltaWHOld = DeltaWHNew;
	end
	TMSE = TMSE / size(training,2);
	Tacc = Tacc / size(training,2) * 100;

	for i = generalization
		Output = feed(i,WeightHidden,WeightOutput);

		v = Output(y(i));
		GMSE = GMSE + (1-v)^2;
		[v,r] = max(Output);
		if y(i) == r
			Gacc++;
		end
	end
	GMSE = GMSE / size(generalization,2);
	Gacc = Gacc / size(generalization,2) * 100;

	printf('TMSE: %f, Tacc: %f, GMSE: %f, Gacc: %f\n',TMSE,Tacc,GMSE,Gacc);
	fflush(stdout);
	if (((TMSE < 0.05)&&(GMSE < 0.05))||((Tacc > 95)&&(Gacc > 95)))
		break
	end
	epoch++;
end

for i = validation
	Output = feed(i,WeightHidden,WeightOutput);

	v = Output(y(i));
	VMSE = VMSE + (1-v)^2;
	[v,r] = max(Output);
	if y(i) == r
		Vacc++;
	end
end
VMSE = VMSE / size(validation,2);
Vacc = Vacc / size(validation,2) * 100;

printf('VMSE: %f, Vacc: %f\n',VMSE,Vacc);
