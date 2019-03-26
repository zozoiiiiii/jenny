/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include <vector>
#include <map>
#include "../parser/ini_parser.h"
#include "../utils/string_util.h"

// namespace
#ifndef NS_JJ_BEGIN
#define NS_JJ_BEGIN                     namespace JJ {
#define NS_JJ_END                       }
#define USING_NS_JJ                     using namespace JJ;
#endif


NS_JJ_BEGIN

class network;


typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    UPSAMPLE,
    REORG,
    BLANK
} LAYER_TYPE;




struct layer
{

    int batch;
    int inputs;
    int outputs;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int size;
    int stride;
    int pad;
    int coords; // box



    int dontload;
    int dontloadscales;





    std::vector<float> biases;      // con, yolo layer, weight file


    float* output;  // con, yolo, maxpool, route, upsample
    std::vector<int8_t> output_int8;    //con, maxpool , route
    size_t workspace_size;
};




struct size_params
{
    int quantized;
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    JJ::network* net;
};


struct network_state
{
    float *truth;
    float *input;
    int8_t *input_int8;
    float *delta;
    float *workspace;
    int train;
    int index;
    JJ::network* net;
};

class ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params) = 0;
    virtual void forward_layer_cpu(network_state state) {}

    layer* getLayer() { return &m_layerInfo; }

    LAYER_TYPE getType() { return m_type; }
    void setType(LAYER_TYPE lt) { m_type = lt; }
protected:
    layer m_layerInfo;
    LAYER_TYPE m_type;
};

NS_JJ_END