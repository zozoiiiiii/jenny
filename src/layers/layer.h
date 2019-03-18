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

// namespace
#ifndef NS_JJ_BEGIN
#define NS_JJ_BEGIN                     namespace JJ {
#define NS_JJ_END                       }
#define USING_NS_JJ                     using namespace JJ;
#endif


NS_JJ_BEGIN

class network;
struct tree
{
    int *leaf;
    int n;
    int *parent;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
};


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
    void(*forward)   (struct layer, struct network_state);
    void(*backward)  (struct layer, struct network_state);
    void(*update)    (struct layer, int, float, float, float);
    void(*forward_gpu)   (struct layer, struct network_state);
    void(*backward_gpu)  (struct layer, struct network_state);
    void(*update_gpu)    (struct layer, int, float, float, float);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int truths;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
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

    int *mask;
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

    tree *softmax_tree;
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

    int *indexes;
    float *rand;
    float *cost;
    std::vector<char> cweights;
    float *state;
    float *prev_state;
    float *forgot_state;
    float *forgot_delta;
    float *state_delta;

    float *concat;
    float *concat_delta;

    std::vector<float> binary_weights;

    char *align_bit_weights_gpu;
    float *mean_arr_gpu;
    float *align_workspace_gpu;
    float *transposed_align_workspace_gpu;
    int align_workspace_size;

    char *align_bit_weights;
    std::vector<float> mean_arr;
    int align_bit_weights_size;
    int lda_align;
    int new_lda;
    int bit_align;

    std::vector<float> biases;
    std::vector<float> biases_quant;
    //float *bias_updates;

    int quantized;

    std::vector<float> scales;
    //float *scale_updates;

    //float *weights;
    std::vector<float> weights;
    //int8_t * weights_int8;
    std::vector<int8_t> weights_int8;


    //float *weight_updates;
    //float *weights_quant_multipler;
    float weights_quant_multipler;
    float input_quant_multipler;

    float *col_image;
    int   * input_layers;
    int   * input_sizes;
    //float * delta;
    //float * output;
    std::vector<float> output;
    //float *output_multipler;
    float output_multipler;
    //int8_t * output_int8;
    std::vector<int8_t> output_int8;
    float * squared;
    float * norms;

    float * spatial_mean;
    std::vector<float>  mean;
    std::vector<float>  variance;

    //float * mean_delta;
    //float * variance_delta;

    std::vector<float> rolling_mean;
    std::vector<float> rolling_variance;

    std::vector<float> x;
    std::vector<float> x_norm;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    float *z_cpu;
    float *r_cpu;
    float *h_cpu;

    std::vector<float> binary_input;

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

class ILayer
{
public:
    virtual bool load(const IniParser* pParser, int section, size_params params) = 0;

    layer m_layerInfo;
};

NS_JJ_END