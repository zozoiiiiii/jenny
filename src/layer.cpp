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