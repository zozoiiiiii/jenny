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

typedef enum {
    SSE, MASKED, SMOOTH
} COST_TYPE;

enum ACTIVATION
{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
};

struct layer
{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    int batch_normalize;

    int batch;
    int inputs;
    int outputs;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int size;
    int stride;
    int reverse;
    int pad;
    int index;  // rout
    int use_bin_output;
    int steps;
    int hidden;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    int focal_loss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;

    std::vector<int> mask;  // yolo layer
    int total;
    float bflops;

    int adam;
    float B1;
    float B2;
    float eps;
    float *m_gpu;
    float *v_gpu;
    int t;
    std::vector<float> m;
    std::vector<float> v;

    int  *map;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    int classfix;
    int absolute;

    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    int* indexes; // maxpoll layer
    float *rand;
    std::vector<float> cost;    // yolo layer
    std::vector<char> cweights;
    float *state;
    float *prev_state;
    float *forgot_state;
    float *forgot_delta;
    float *state_delta;

    float *concat;
    float *concat_delta;

    char *align_bit_weights_gpu;
    float *mean_arr_gpu;
    float *align_workspace_gpu;
    float *transposed_align_workspace_gpu;
    int align_workspace_size;


    std::vector<float> biases;      // con, yolo layer, weight file


    std::vector<int> input_layers; // route
    std::vector<int> input_sizes; // route 
    float* output;  // con, yolo, maxpool, route, upsample
    float output_multipler;
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

    int entry_index(int batch, int location, int entry);
protected:
    layer m_layerInfo;
};

NS_JJ_END