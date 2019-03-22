/************************************************************************/
/*
@author:  junliang
@brief:   convolution layer
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "layer.h"
#include "network.h"
#include <intrin.h> 

NS_JJ_BEGIN



struct ConvolutionalLayerInfo
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

    ConvolutionalLayerInfo() :filters(1), size(1), stride(1), pad(0), padding(0), activation(LOGISTIC), batch_normalize(0),
        binary(0), xnor(0), bin_output(0), flipped(0), dot(0) {}
};


struct ConvolutionWeight
{
    float* biases;
    float* scales;
    float* rolling_mean;
    float* rolling_variance;
    float *weights;

    // tempory
    float* binary_weights;
    float *binary_input;
};






typedef layer convolutional_layer;
class ConvolutionLayer : public ILayer
{
public:
    static void activate_array(float *x, const int n, const ACTIVATION a);
public:
    virtual bool load(const IniParser* pParser, int section, size_params params);
    virtual void forward_layer_cpu(network_state state);
    ConvolutionWeight* getWeight() { return &m_weight; }

    void load_convolutional_weights_cpu(FILE *fp);
    void binary_align_weights();
private:
    layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation,
        int batch_normalize, int binary, int xnor, int adam, int quantized, int use_bin_output);



private:
    ConvolutionWeight m_weight;
};

NS_JJ_END