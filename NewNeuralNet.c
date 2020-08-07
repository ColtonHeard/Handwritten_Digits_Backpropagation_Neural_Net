#ifndef STANDARD
#define STANDARD
#include <stdlib.h>
#include <stdio.h>
#endif

//#include "leakcheck.h"
#include "new_network.h"

#define BATCH_SIZE 4000
#define CHECK_SIZE 4000

double sigmoidVal(unsigned char c)
{
    double f = (double) c;
    return 1.0/(1.0 + exp(-f));
}

void printMatrix(Matrix * m)
{
    for (int row = 0; row < m->height; row++)
    {
        printf("[");
        for (int col = 0; col < m->width; col++)
        {
            if (col != m->width - 1)
            {
                printf("%lf, ", m->values[(m->width * row) + col]);
            }
            else
            {
                printf("%lf]\n", m->values[(m->width * row) + col]);
            }
            
        }
    }
    printf("\n");
}

int main()
{
    // printf("Initalizing checker...\n");
    // init_checker();
    // printf("Checker initialized.\n");

    FILE * image_file = fopen("train-images.idx3-ubyte", "rb");
    FILE * label_file = fopen("train-labels.idx1-ubyte", "rb");
    // FILE * image_file = fopen("C:\\Users\\wwwco\\Desktop\\CProjects\\Gradient Descent\\train-images.idx3-ubyte", "rb");
    // FILE * label_file = fopen("C:\\Users\\wwwco\\Desktop\\CProjects\\Gradient Descent\\train-labels.idx1-ubyte", "rb");

    if (image_file == NULL)
    {
        printf("Images file could not be opened!\n");
        return -1;
    }
    else if (label_file == NULL)
    {
        printf("Labels file could not be opened!\n");
        return -1;
    }

    //Contains space for the pixels of all 60,000 training images + 16 bytes of information at the beginning of the file
    unsigned char * pixels = calloc(47040016, sizeof(unsigned char));

    //Contains space for the 60,000 labels whose values should be between 0-9 + the 8 bytes of information at the beginning of the file
    unsigned char * labels = calloc(60008, sizeof(unsigned char));

    //Read and store the image pixels and labels
    fread(pixels, sizeof(unsigned char), 47040016, image_file);
    fread(labels, sizeof(unsigned char), 60008, label_file);

    //Close the data files
    fclose(image_file);
    fclose(label_file);

    printf("This value should be 28: %u\n", pixels[11]);

    //Setup Neural Net
    // printf("net\n");
    Network * n = (Network*) calloc(1, sizeof(Network));
    int* f = (int*) calloc(4, sizeof(int));
    *f = 784;
    *(f + 1) = 16;
    *(f + 2) = 16;
    *(f + 3) = 10;
    init_network(n, 4, f, 0.05f);
    // printf("done net\n");

    //Start running through the training data
    int correctCount = 0;
    int totalTested = 0;
    int correctThisRun = 0;
    for (int z = 0; z < 35000; z++)
    {
        for (int i = 0; i < 60000; i++)
        {
            if (totalTested == 0)
            {
                // printMatrix(n->weights + 2);
            }

            resetActivation(n);

            // printf("Creating input matrix\n");
            Matrix * input = calloc(1, sizeof(Matrix));
            initMatrix(input, 1, 784);

            Matrix * expected = calloc(1, sizeof(Matrix));
            initMatrix(expected, 1, 10);
            expected->values[labels[8 + i]] = 1.0;
            n->expected = expected;

            for (int x = 0; x < 784; x++)
            {
                (input->values)[x] = pixels[16 + (784 * i) + x] / 255.0;
                // checkIn->values[x] = pixels[16 + (784 * i) + x] / 255.0;
            }
            n->input = input;

            feedforward(n);

            //Check output and display
            int check = 0;
            if (i % CHECK_SIZE == 0)
            {
                check = 1;
                printf("Network has gotten %d out of %d correct\n", correctCount, totalTested);
                printf("Correct to Error: %lf\n", ((double) correctThisRun) / CHECK_SIZE);
                // printMatrix(n->output);
                correctThisRun = 0;
            }

            int out = findOutput(n);
            totalTested++;
            if (out == labels[8 + i])
            {
                correctCount++;
                correctThisRun++;
            }
            if (check)
            {
                printf("Network guessed %d, was %u\n\n", out, labels[8 + i]);
            }

            //BACKPROPOGATE
            if (i % CHECK_SIZE == 0)
            {
                backprop(n, 0);
            }
            else
            {
                backprop(n, 0);
            }
            
            free(input->values);
            free(input);
            free(expected->values);
            free(expected);

            // printf("reseting activation\n");
        }
    }
    printf("Network has gotten %d out of %d correct\n", correctCount, totalTested);
    // printMatrix(n->weights + 2);

    free(pixels);
    free(labels);

    free(f);

    for (int i = 0; i < n->layers; i++)
    {
        if (i < n->layers - 1)
        {
            free((n->weights + i)->values);
            free((n->biases + i)->values);
            free((n->activationValues + i)->values);
        }
        else
        {
            free((n->activationValues + i)->values);
        }
        
    }

    free(n->weights);
    free(n->biases);
    free(n->activationValues);
    free(n);

    // deinit_checker();

    return 1;
}