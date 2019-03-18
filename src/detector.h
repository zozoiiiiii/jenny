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


NS_JJ_BEGIN
class Detector
{
public:
    void loadData(const char* cfgfile, const char* weightsfile);
    void detectImage(char **names, char *filename, float thresh, int quantized, int dont_show);

private:

    JJ::network* readConfigFile(const char* cfgFile, int batch, int quantized);
    bool parseNetOptions(const IniParser* pIniParser, JJ::network* pNetWork);
    void splitString(std::vector<std::string>& result, const std::string& str, const std::string& delims);
    void splitInt(std::vector<int>& result, const std::string& str, const std::string& delims);
    void splitFloat(std::vector<float>& result, const std::string& str, const std::string& delims);
    std::string Trim(const std::string& str, const char ch);

    JJ::learning_rate_policy get_policy(const char *s);
    JJ::LAYER_TYPE string_to_layer_type(const std::string& type);


    //convolutional_layer parse_convolutional(const IniParser* pParser, int section, size_params params);
    //layer parse_region(list *options, size_params params);
    //int *parse_yolo_mask(char *a, int *num);
    //layer parse_yolo(list *options, size_params params);
    //softmax_layer parse_softmax(list *options, size_params params);
    //maxpool_layer parse_maxpool(list *options, size_params params);
    //layer parse_reorg(list *options, size_params params);
    //layer parse_upsample(list *options, size_params params, network net);
    //layer parse_shortcut(list *options, size_params params, network net);
    //route_layer parse_route(list *options, size_params params, network net);

private:
    JJ::network* m_pNet;
};
NS_JJ_END