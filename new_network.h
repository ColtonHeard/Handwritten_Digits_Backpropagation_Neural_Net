#ifndef STANDARD
#define STANDARD
#include <stdlib.h>
#include <stdio.h>
#endif

#include <time.h>
#include <math.h>

typedef struct matrix
{
    int width;
    int height;
    double* values;
} Matrix;

typedef struct network
{
    //The first matrix in this list represents the weight of the connections from the input layer to the second layer, the next being from the second layer to the third, etc.
    Matrix * weights;
    
    //The first matrix in this list represents the biases for the 2nd layer of nodes (aka the first non-input layer)
    Matrix * biases;

    Matrix * activationValues;

    Matrix * input;
    Matrix * output;
    Matrix * expected;

    float learningRate;

    int layers;
    int * layerSizes;
} Network;

void printNewNetMatrix(Matrix * m)
{
    for (int row = 0; row < m->height; row++)
    {
        printf("[");
        for (int col = 0; col < m->width; col++)
        {
            if (col != m->width - 1)
            {
                printf("%f, ", m->values[(m->width * row) + col]);
            }
            else
            {
                printf("%f]\n", m->values[(m->width * row) + col]);
            }
            
        }
    }
    printf("\n");
}

void initMatrix(Matrix * m, int width, int height)
{
    m->width = width;
    m->height = height;
    m->values = (double*) calloc(height * width, sizeof(double));
}

//Creates a seperate non-shallow copy of matrix m
Matrix* copyMatrix(Matrix * m)
{
    Matrix* mat = (Matrix*) calloc(1, sizeof(Matrix));
    initMatrix(mat, m->width, m->height);

    for (int i = 0; i < (m->height * m->width); i++)
    {
        mat->values[i] = m->values[i];
    }

    return mat;
}

//Initiates the network with layerCount layers (minimum of two should be used), with each layers size specified in the list layer_sizes
//Creates layerCount - 1 weight and bias matrices and adds them to the networks weights and biases lists with random initial values

//Calloc in large chunks
void init_network(Network * n, int layerCount, int * layer_sizes, float learning_rate)
{
    n->layers = layerCount;
    n->layerSizes = layer_sizes;
    n->learningRate = learning_rate;
    n->weights = (Matrix *) calloc(layerCount - 1, sizeof(Matrix));
    n->biases = (Matrix *) calloc(layerCount - 1, sizeof(Matrix));
    n->activationValues = (Matrix *) calloc(layerCount, sizeof(Matrix));

    //Create weight matrices with random initial values
    for (int i = 0; i < layerCount - 1; i++)
    {
        int from = layer_sizes[i];
        int to = layer_sizes[i + 1];

        //Create weight matrix of with a height that's the same as the number of nodes in the first layer and a width the same as the number of nodes in the next layer
        //The weight of a connection from node a in the first layer to node b in the second is the value at index (b, a) where (row, col)
        initMatrix(n->weights + i, from, to);

        //Set random generator seed and fill matrix with values
        srand(time(0));
        for (int x = 0; x < from * to; x++)
        {
            (n->weights + i)->values[x] = ((double) rand()) / ((double) RAND_MAX);
            // (n->weights + i)->values[x] = 0.0;
        }
    }

    //Create bias matrices with random initial values
    for (int i = 1; i < layerCount; i++)
    {
        initMatrix(n->biases + (i - 1), 1, layer_sizes[i]);

        //Fill matrix with values
        srand(time(0));
        for (int x = 0; x < layer_sizes[i]; x++)
        {
            (n->biases + (i - 1))->values[x] = ((double) rand()) / ((double) RAND_MAX);
            // (n->biases + (i - 1))->values[x] = 0.0;
        }
    }
}

void resetActivation(Network * n)
{
    // printf("Starting activation reset...\n");
    for (int i = 0; i < n->layers; i++)
    {
        // printf("Freeing values in section %d of activations\n", i);
        free((n->activationValues + i)->values);
    }
    free(n->activationValues);

    n->activationValues = (Matrix*) calloc(n->layers, sizeof(Matrix));
    for (int i = 0; i < n->layers; i++)
    {
        (n->activationValues + i)->width = 1;
        (n->activationValues + i)->height = n->layerSizes[i];
    }
}

//Applies the sigmoid function element-wise on all values in Matrix m
Matrix* sigmoid(Matrix * m)
{
    for (int i = 0; i < m->width * m->height; i++)
    {
        (m->values)[i] = 1.0/( 1.0 + exp(-((m->values)[i])) );
    }

    return m;
}

void feedforward(Network * n)
{
    Matrix * in = copyMatrix(n->input);
    (n->activationValues)->values = in->values;
    free(in);

    for (int i = 0; i < n->layers - 1; i++)
    {
        Matrix * baseLayerValues = (Matrix*) calloc(1, sizeof(Matrix));
        initMatrix(baseLayerValues, 1, n->layerSizes[i + 1]);

        for (int row = 0; row < (n->weights + i)->height; row++)
        {
            double val = 0.0;

            for (int col = 0; col < (n->weights + i)->width; col++)
            {
                double weight = (n->weights + i)->values[(row * (n->weights + i)->width) + col];
                double actVal = (n->activationValues + i)->values[col];
                // printf("%lf * %lf = %lf\n", weight, actVal, weight * actVal);
                val += weight * actVal;
            }

            val += (n->biases + i)->values[row];

            baseLayerValues->values[row] = val;
        }

        (n->activationValues + (i + 1))->values = sigmoid(baseLayerValues)->values;
        free(baseLayerValues);
    }

    n->output = (n->activationValues + (n->layers - 1));
}

//Recursive function that calculates the gradients for the given layers and all layers before it
void calculateGradientForLayer(Network * n, int layer, Matrix * e)
{
    if (layer == 0)
        return;

    //Calculate partial derivative with respect to the activation of node k
    for (int k = 0; k < n->layerSizes[layer]; k++)
    {
        double sum = 0.0;

        for (int j = 0; j < n->layerSizes[layer + 1]; j++)
        {
            //The weight of an connection from node a in the first layer to node b in the second is the value at index (b, a) where (row, col)
            double w_jk = (n->weights + layer)->values[((n->weights + layer)->width * j) + k];
            double actJ = (n->activationValues + layer + 1)->values[j];
            double sigDir = actJ * (1.0 - actJ);
            double e_j = (e + layer + 1)->values[j];

            double a_k = w_jk * sigDir * e_j;
            sum += a_k;
        }

        (e + layer)->values[k] = sum;
    }

    calculateGradientForLayer(n, layer - 1, e);
}

//errorSum represents the matrix containing all the costs/errors of the networks outputs added together
//Designed to be done after every sample
//8kb mem leak during, 12 and 16 swapping between
//void backprop(Network * n, Matrix * errorSum, int debug)
void backprop(Network * n, int debug)
{
    Matrix * e = (Matrix*) calloc(n->layers, sizeof(Matrix));

    for (int i = 0; i < n->layers; i++)
    {
        initMatrix((e + i), 1, n->layerSizes[i]);
    }

    //printf("\tCalculating output errors...\n");
    //Calculate output errors
    for (int i = 0; i < n->output->height; i++)
    {
        //printf("Grabbing activation\n");
        float oj = (n->activationValues + n->layers - 1)->values[i];
        float dir = oj * (1 - oj);
        (e + n->layers - 1)->values[i] = dir * (oj - n->expected->values[i]);
    }

    if (debug)
    {
        printNewNetMatrix((e + n->layers - 1));
    }

    //printf("\tCalculating hidden errors...\n");
    //Start calculating hidden errors
    // for (int x = n->layers - 2; x > 0; x--)
    // {
    //     if (debug)
    //         printf("Calculating errors for nodes in layer %d\n", x);
    //     //Calculate error for each node in current layer
    //     for (int i = 0; i < (e + x)->height; i++)
    //     {
    //         //printf("Calculating errors for node %d in layer %d\n", i, x);
    //         float act = (n->activationValues + x)->values[i];
    //         float errSum = 0.0;

    //         //printf("Finding error sum\n");
    //         for (int z = 0; z < (e + x + 1)->height; z++)
    //         {
    //             float weightVal = (n->weights + x)->values[(z * ((n->weights + x)->width)) + i];

    //             float nextErr = (e + x + 1)->values[z];

    //             errSum += weightVal * nextErr;
    //         }

    //         //printf("Calculating error for node %d\n", i);
    //         float err = act * (1 - act) * errSum;
    //         (e + x)->values[i] = err;
    //     }
    // }

    calculateGradientForLayer(n, n->layers - 2, e);

    // float learn = 0.0;
    // for (int i = 0; i < n->expected->height; i++)
    // {
    //     learn += n->expected->values[i] - n->output->values[i];
    // }
    // learn /= n->expected->height;
    // learn = learn * n->learningRate;
    // learn = abs(learn);

    //Apply weight changes based on errors
    for (int weightLayer = 0; weightLayer < n->layers - 1; weightLayer++)
    {
        for (int k = 0; k < (n->weights + weightLayer)->width; k++)
        {
            for (int j = 0; j < (n->weights + weightLayer)->height; j++)
            {
                //Multiply by sigDir?
                // double act_j  = (n->activationValues + weightLayer + 1)->values[j];
                // double sigDir = act_j * (1 - act_j);

                double a_j = (e + weightLayer + 1)->values[j];
                double a_k = (n->activationValues + weightLayer)->values[k];
                double gradient = a_j * a_k; // * sigDir;
                (n->weights + weightLayer)->values[((n->weights + weightLayer)->width * j) + k] += (-(n->learningRate)) * gradient;
            }
        }
    }

    //Apply weight changes based on errors
    // for (int i = 0; i < n->layers - 1; i++)
    // {
    //     for (int y = 0; y < (n->weights + i)->height; y++)
    //     {
    //         for (int x = 0; x < (n->weights + i)->width; x++)
    //         {
    //             (n->weights + i)->values[(y * (n->weights + i)->width) + x] += (-(learn)) * (e + i + 1)->values[y] * (n->activationValues + i)->values[x];
    //             // (n->weights + i)->values[(y * (n->weights + i)->width) + x] += (-(n->learningRate)) * (e + i + 1)->values[y] * (n->activationValues + i)->values[x];
    //         }
    //     }
    // }

    //Apply bias changes 
    for (int biasLayer = 0; biasLayer < n->layers - 1; biasLayer++)
    {
        for (int bias = 0; bias < n->layerSizes[biasLayer + 1]; bias++)
        {
            double act_j = (n->activationValues + biasLayer + 1)->values[bias];
            double sigDir = act_j * (1 - act_j);
            double a_j = (e + biasLayer + 1)->values[bias];
            double gradient = a_j * sigDir;

            (n->biases + biasLayer)->values[bias] += (-(n->learningRate)) * gradient;
        }
    }

    // Apply bias changes
    // for (int i = 0; i < n->layers - 1; i++)
    // {
    //     for (int x = 0; x < n->layerSizes[i + 1]; x++)
    //     {
    //         //printf("Bias error for layer %d node %d = %lf\n", i, x, (e + i + 1)->values[x]);
    //         // printf("Bias gradient for bias %d in layer %d is %lf\n", x, i + 1, (e + i + 1)->values[x]);
    //         (n->biases + i)->values[x] +=  (-learn) * (e + i + 1)->values[x];
    //     }
    // }

    //free values
    for (int i = 0; i < n->layers; i++)
    {
        free((e + i)->values);
    }
    free(e);
}

//Returns the highest scoring output of the network
int findOutput(Network * n)
{
    int i = 0;
    double greatest = n->output->values[0];
    for (int x = 0; x < n->output->height; x++)
    {
        //printf("Value of output at %d is %f, greatest is %f\n", x, n->output->values[x], greatest);
        if (n->output->values[x] > greatest)
        {
            greatest = n->output->values[x];
            i = x;
        }
    }

    return i;
}