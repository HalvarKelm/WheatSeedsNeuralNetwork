#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "WheatHelper.h"

#define DATATOREAD 210
#define EPOCHS 50000
#define LEARNINGRATE 0.1

//sigmoid activation function
//calculate the sigmoid for a given double
float sigmoid(float x){
    return (1.0/(1.0+(exp(-1.0 * x))));
}

//calculate the sigmoid prime for a given double
float sigmoid_prime(float x){
    return sigmoid(x) * (1.0 - sigmoid(x));
}

//index of highest value from an array
int getIndexOfHighestValue(float *array,int size){
	int index = 0;
	for(int i = 1; i < size; i++){
    	if(array[i] > array[index]){
    	    index = i;
    	}
	}
	return index;
}

//check the accuracy of the network
int evaulate(DATASET *trainData, float *weightsFromInputToHidden, float *weightsFromHiddenToOutput,float *biasesOfHidden, float *biasesOfOutput, int INPUTLAYERNODES, int HIDDENLAYERNODES, int OUTPUTLAYERNODES) {
	float cost = 0;
	int right = 0;
	for (int i = 0; i < DATATOREAD; i++) {
		        //backpropagate to get the nabla changes of every weight and bias in the whole network
				//backpropagation for the weights of the output layer
				//setting up NNs first layer
				float input[INPUTLAYERNODES];
				int node = 0;
				for (int j = 0; j < INPUTLAYERNODES; j++) {
					input[node++] = trainData[i].value[j];
				}

				//setting up NNs fist hidden layer
				float hidden[HIDDENLAYERNODES];
	for (int j = 0; j < HIDDENLAYERNODES; j++) {
		//calculate NNs hidden layer
		float z = 0;
		for (int k = 0; k < INPUTLAYERNODES; k++) {
			z += input[k] * weightsFromInputToHidden[j*INPUTLAYERNODES+k];
		}
		z += biasesOfHidden[j];
		hidden[j] = z;
	}

    //setting up the output layer
	float output[OUTPUTLAYERNODES];
	for (int j = 0; j < OUTPUTLAYERNODES; j++) {
		//calculate NNs output layer
		float z = 0;
		for (int k = 0; k < HIDDENLAYERNODES; k++) {
			z += sigmoid(hidden[k]) * weightsFromHiddenToOutput[j*HIDDENLAYERNODES+k];
		}
		z += biasesOfOutput[j];
		output[j] = sigmoid(z);
	}
				//now outputs are my z(L) in the output layer
				//inputs are of course my pixel in the input layer
				//hidden1 are my nodes in the h1 layer and hidden2 my nodes in the h2 layer
				int target[OUTPUTLAYERNODES];
				for(int j = 0; j < OUTPUTLAYERNODES; j++){
					if(trainData[i].value[7] == j){
						target[j] = 1;
					}else{
						target[j] = 0;
					}
				}
        float toCost = 0;
		for(int x = 0; x < OUTPUTLAYERNODES; x++){
			toCost += pow((sigmoid(output[x]) - target[x]),2);
		}
        cost += toCost;
		//printf("The cost this time is %f\n", toCost);

		if(trainData[i].value[7] == (float)getIndexOfHighestValue(output,3)){
			right++;
		}else{
		//	printf("GUESS %f - %f RIGHT\n", (float)getIndexOfHighestValue(output,3), trainData[i].value[7]);
			for(int i = 0; i < OUTPUTLAYERNODES; i++){
		//		printf("OUTPUT%i : %f\n", i, output[i]);
			}
		}
	}
//	printf("I got %f average cost in the last %i trainData.\n", ((float)cost/(float)DATATOREAD), DATATOREAD);
	return right;
}

void updateWeights(float *input, float *hidden, float *deltaHidden, float *deltaOutput, float *weightsFromInputToHidden, float *weightsFromHiddenToOutput, float *biasesOfHidden, float *biasesOfOutput, int INPUTLAYERNODES, int HIDDENLAYERNODES, int OUTPUTLAYERNODES){
    //update the last layer
	int nodeOfHidden = 0;
	int nodeOfOutput = 0;
	for(int i = 0; i < HIDDENLAYERNODES * OUTPUTLAYERNODES; i++){
        nodeOfHidden = i%HIDDENLAYERNODES;
        if(i%HIDDENLAYERNODES == 0 && i != 0){
			nodeOfOutput++;
		}
		weightsFromHiddenToOutput[i] += LEARNINGRATE * (deltaOutput[nodeOfOutput] * sigmoid(hidden[nodeOfHidden]));
	}
	for(int i = 0; i < OUTPUTLAYERNODES; i++){
		biasesOfOutput[i] += LEARNINGRATE * deltaOutput[i];
	}
	//update the hidden layer
	nodeOfHidden = 0;
	int nodeOfInput = 0;
	for(int i = 0; i < HIDDENLAYERNODES * INPUTLAYERNODES; i++){
        nodeOfInput = i%INPUTLAYERNODES;
        if(i%INPUTLAYERNODES == 0 && i != 0){
	    	nodeOfHidden++;
		}
		weightsFromInputToHidden[i] += LEARNINGRATE * (deltaHidden[nodeOfHidden] * input[nodeOfInput]);
	}
	for(int i = 0; i < HIDDENLAYERNODES; i++){
    	biasesOfHidden[i] += LEARNINGRATE * deltaHidden[i];
	}
}

void backPropagateError(float *input, float *hidden, float *output, float *target, float *deltaHidden, float *deltaOutput, float *weightsFromHiddenToOutput, int HIDDENLAYERNODES, int OUTPUTLAYERNODES){
    //error of last layer
    float errorOutput[OUTPUTLAYERNODES];
	for(int i = 0; i < OUTPUTLAYERNODES; i++){
        errorOutput[i] = (target[i] - sigmoid(output[i]));
    }

    for(int i = 0; i < OUTPUTLAYERNODES; i++){ 
		deltaOutput[i] = errorOutput[i] * sigmoid_prime(output[i]);
	}

	//error of hidden layer
	float errorHidden[HIDDENLAYERNODES];
	for(int i = 0; i < HIDDENLAYERNODES; i++){
		errorHidden[i] = 0.0;
		for(int j = 0; j < OUTPUTLAYERNODES; j++){
			errorHidden[i] += weightsFromHiddenToOutput[j*OUTPUTLAYERNODES+i] * deltaOutput[j];
		}
	}
	for(int i = 0; i < HIDDENLAYERNODES; i++){
		deltaHidden[i] = errorHidden[i] * sigmoid_prime(hidden[i]);
    }
}

float* feedForward(float *input, float* hidden, float* output, DATASET *trainData,int data,float *weightsFromInputToHidden,float *weightsFromHiddenToOutput,float *biasesOfHidden, float *biasesOfOutput, int INPUTLAYERNODES,int HIDDENLAYERNODES,int OUTPUTLAYERNODES){
    for (int j = 0; j < INPUTLAYERNODES; j++) {
		input[j] = trainData[data].value[j];
	}
    //setting up NNs fist hidden layer
	for (int j = 0; j < HIDDENLAYERNODES; j++) {
		//calculate NNs hidden layer
		float z = 0;
		for (int k = 0; k < INPUTLAYERNODES; k++) {
			z += input[k] * weightsFromInputToHidden[j*INPUTLAYERNODES+k];
		}
		z += biasesOfHidden[j];
		hidden[j] = z;
	}

    //setting up the output layer
	for (int j = 0; j < OUTPUTLAYERNODES; j++) {
		//calculate NNs output layer
		float z = 0;
		for (int k = 0; k < HIDDENLAYERNODES; k++) {
			z += sigmoid(hidden[k]) * weightsFromHiddenToOutput[j*HIDDENLAYERNODES+k];
		}
		z += biasesOfOutput[j];
		output[j] = z;
	}
    return output;
}

//set a dataset to values between 1 and 0
void normalize(DATASET trainData[]){
	float highestValue = trainData[0].value[0];
	for(int i = 0; i < DATATOREAD; i++){
		for(int j = 0; j < 7; j++){
			if(highestValue < trainData[i].value[j]){
				highestValue = trainData[i].value[j];
			}
		}
	}
	for(int i = 0; i < DATATOREAD; i++){
		for(int j = 0; j < 7; j++){
			trainData[i].value[j] = trainData[i].value[j] / highestValue;
		}
	}
}

int predict(float* data,float * weightsFromInputToHidden,float * weightsFromHiddenToOutput,float * biasesOfHidden,float * biasesOfOutput,int INPUTLAYERNODES,int HIDDENLAYERNODES,int OUTPUTLAYERNODES){
    float input[INPUTLAYERNODES];
    for (int j = 0; j < INPUTLAYERNODES; j++) {
		input[j] = data[j];
	}

    //setting up NNs fist hidden layer
    float hidden[HIDDENLAYERNODES];
	for (int j = 0; j < HIDDENLAYERNODES; j++) {
		//calculate NNs hidden layer
		float z = 0;
		for (int k = 0; k < INPUTLAYERNODES; k++) {
			z += input[k] * weightsFromInputToHidden[j*INPUTLAYERNODES+k];
		}
		z += biasesOfHidden[j];
		hidden[j] = z;
	}

    //setting up the output layer
    float output[OUTPUTLAYERNODES];
	for (int j = 0; j < OUTPUTLAYERNODES; j++) {
		//calculate NNs output layer
		float z = 0;
		for (int k = 0; k < HIDDENLAYERNODES; k++) {
			z += sigmoid(hidden[k]) * weightsFromHiddenToOutput[j*HIDDENLAYERNODES+k];
		}
		z += biasesOfOutput[j];
		output[j] = z;
		printf("For %i - %f\n",j, sigmoid(output[j]));
	}
	for (int j = 0; j < OUTPUTLAYERNODES; j++) {
		output[j] = sigmoid(output[j]);
	}
    printf("It should be %f\n", data[INPUTLAYERNODES]);
	if(getIndexOfHighestValue(output, OUTPUTLAYERNODES) == (int)data[7]){
		return 1;
	}else{
		return 0;
	}
}

void trainNetwork(DATASET *trainData, float *weightsFromInputToHidden,float *weightsFromHiddenToOutput,float *biasesOfHidden,float *biasesOfOutput, int INPUTLAYERNODES, int HIDDENLAYERNODES, int OUTPUTLAYERNODES){
    for (int epoch = 0; epoch < EPOCHS; epoch++){
        float error = 0;
        for(int train = 0; train < DATATOREAD; train++){
            float input[INPUTLAYERNODES];
            float hidden[HIDDENLAYERNODES];
            float output[OUTPUTLAYERNODES];
            float target[OUTPUTLAYERNODES];
            for(int j = 0; j < OUTPUTLAYERNODES; j++){
			    if(trainData[train].value[7] == j){
			    	target[j] = 1;
			    }else{
				    target[j] = 0;
			    }
		    }
            feedForward(input, hidden, output, trainData, train, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput, INPUTLAYERNODES, HIDDENLAYERNODES, OUTPUTLAYERNODES);
            for(int i = 0; i < OUTPUTLAYERNODES; i++){
                error += pow(target[i] - sigmoid(output[i]),2);
            }
            float deltaHidden[HIDDENLAYERNODES];
            float deltaOutput[OUTPUTLAYERNODES];
            backPropagateError(input, hidden, output, target, deltaHidden, deltaOutput, weightsFromHiddenToOutput, HIDDENLAYERNODES, OUTPUTLAYERNODES);
            updateWeights(input, hidden, deltaHidden, deltaOutput,weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput, INPUTLAYERNODES, HIDDENLAYERNODES, OUTPUTLAYERNODES);
        }
        printf("EPOCH %i | error = %f\n",epoch, error);
		//int q = evaulate(trainData, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput, INPUTLAYERNODES, HIDDENLAYERNODES, OUTPUTLAYERNODES);
		printf("Epoch %i: %i\n", epoch, DATATOREAD);
    }
}

int main(int argc, char* argv[]){
    //random time for random values later for weights and biases
	srand(time( NULL));

    //load the input files
	FILE* trainingData = fopen(argv[1], "r");
    if(argc < 4){
        printf("Not enough arguments\n");
        return 1;
    }

    if(trainingData == NULL){
        printf("File not found!");
        return 1;
    }

    int INPUTLAYERNODES = atoi(argv[2]);
    int HIDDENLAYERNODES = atoi(argv[3]);
    int OUTPUTLAYERNODES = atoi(argv[4]);

    DATASET trainData[DATATOREAD];

    char line[127];
    int lineCount = 0;
    while (fgets ( line, sizeof line, trainingData )!= NULL && lineCount < DATATOREAD){
        int currentValueCount = 0;
        char currentValue[10];
        for(int i = 0; i < 10; i++){ currentValue[i] ==(char) 0;}
        int currentValuePos = 0;
        for(int i = 0; i < sizeof(line); i++){
            if(line[i] != ',' && line[i] != '\n'){
                currentValue[currentValuePos] = line[i];
                currentValuePos++;
            }else{
                trainData[lineCount].value[currentValueCount] = atof(currentValue);
                for(int j = 0; j < 10; j++){
                    currentValue[j] = (char) 0;
                }
                currentValueCount++;
                currentValuePos = 0;
            }
        }
        lineCount++;
    }
	normalize(trainData);
    for(int i = 0; i < DATATOREAD; i++){
        trainData[i].value[7] -= 1;
    }

    for(int i = 0; i < DATATOREAD; i++){
        printf("Data %i | ", i);
        for(int j = 0; j < INPUTLAYERNODES+1; j++){
            printf("%f|",trainData[i].value[j]);
        }
        printf("\n");
    }

	//setting up the network
	float weightsFromInputToHidden[INPUTLAYERNODES * HIDDENLAYERNODES];
	for (int i = 0; i < INPUTLAYERNODES * HIDDENLAYERNODES; i++) {
		weightsFromInputToHidden[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	float weightsFromHiddenToOutput[HIDDENLAYERNODES * OUTPUTLAYERNODES];
	for (int i = 0; i < HIDDENLAYERNODES * OUTPUTLAYERNODES; i++) {
		weightsFromHiddenToOutput[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	float biasesOfHidden[HIDDENLAYERNODES];
	for (int i = 0; i < HIDDENLAYERNODES; i++) {
		biasesOfHidden[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	float biasesOfOutput[OUTPUTLAYERNODES];
	for (int i = 0; i < OUTPUTLAYERNODES; i++) {
		biasesOfOutput[i] = (float) rand() / RAND_MAX * 2.0 - 1.0;
	}
	trainNetwork(trainData, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput, INPUTLAYERNODES, HIDDENLAYERNODES, OUTPUTLAYERNODES);
    fclose(trainingData);
	printf("predicting...\n");
	int right = 0;
    for(int i = 0; i < DATATOREAD; i++){
        float data[INPUTLAYERNODES+1];
        data[0] = trainData[i].value[0];
        data[1] = trainData[i].value[1];
		data[2] = trainData[i].value[2];
        data[3] = trainData[i].value[3];
		data[4] = trainData[i].value[4];
		data[5] = trainData[i].value[5];
		data[6] = trainData[i].value[6];
        data[7] = trainData[i].value[7];
        right += predict(data, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesOfHidden, biasesOfOutput, INPUTLAYERNODES, HIDDENLAYERNODES, OUTPUTLAYERNODES);
    }
	printf("I got %i / %i\n", right, DATATOREAD);
    printf("end.\n");
    return 0;
}