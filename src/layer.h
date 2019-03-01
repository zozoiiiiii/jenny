/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>

typedef enum {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

struct ConvolutionalLayer
{
    int filters;
    int size;
    int stride;
    int pad;
    int padding;
    ACTIVATION activation;

    int batch_normalize;
    int binary;
    int xnor;
    int bin_output;

    int flipped;
    float dot;

    ConvolutionalLayer() :filters(1), size(1), stride(1), pad(0), padding(0), activation(LOGISTIC), batch_normalize(0),
        binary(0), xnor(0), bin_output(0), flipped(0), dot(0) {}
};


struct network
{
    int quantized;
    float *workspace;
    int n;  // layer count
    int batch;
    float *input_calibration;
    int input_calibration_size;
    uint64_t *seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int h, w, c;
    int max_crop;
    int min_crop;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;

    int gpu_index;
    tree *hierarchy;
    int do_input_calibration;
};