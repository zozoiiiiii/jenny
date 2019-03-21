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
#include "layer.h"



NS_JJ_BEGIN



enum learning_rate_policy
{
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
};

struct network
{
    int quantized;
    std::vector<float> workspace;
    int n;  // layer count

    int batch;
    std::vector<float> input_calibration;
    //int input_calibration_size;
    uint64_t *seen;                 // weight file
    //float epoch;
    int subdivisions;
    float momentum;
    float decay;
    //layer *layers;
    //std::vector<layer> layers;
    std::vector<ILayer*> jjLayers;

    int outputs;
    std::vector<float> output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    std::vector<float> scales;
    std::vector<int> steps;
    //int num_steps;    // get from vector
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

    //int gpu_index;
    //tree *hierarchy;
    int do_input_calibration;
};



class NetWork
{
public:
    static int get_network_output_size(network* pNet);
    static std::vector<float> get_network_output(network* pNet);
};
NS_JJ_END