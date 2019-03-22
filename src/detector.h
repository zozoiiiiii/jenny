/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>
#include "layers/convolutional_layer.h"
#include "parser/ini_parser.h"
#include "utils/string_util.h"
#include "box.h"

#include "utils/image_util.h"

NS_JJ_BEGIN

struct detection_with_class
{
    detection det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
};

class Detector
{
public:
    static Detector* instance();
    void detectImage(char **names, char *cfgfile, char *weightfile, char *filename, float thresh, int quantized, int dont_show);

private:
    JJ::network* readConfigFile(const char* cfgFile, int batch, int quantized);
    bool readWeightFile(network *net, char *filename, int cutoff);

    bool parseNetOptions(const IniParser* pIniParser, JJ::network* pNetWork);
    

    JJ::learning_rate_policy get_policy(const char *s);
    JJ::LAYER_TYPE string_to_layer_type(const std::string& type);



    detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num);


    void draw_detections_v3(ImageInfo im, detection *dets, int num, float thresh, char **names, ImageInfo **alphabet, int classes, int ext_output);



    void yolov2_fuse_conv_batchnorm(network* net);

    void calculate_binary_weights(network* net);



    void yolov2_forward_network_cpu(network* net, network_state state);
    float* network_predict_cpu(network* net, float *input);


    void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter);
    detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);



private:
    JJ::network* m_pNet;
};
NS_JJ_END