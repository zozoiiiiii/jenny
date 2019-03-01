/************************************************************************/
/*
@author:  junliang
@brief:   
@time:    2019/02/22
*/
/************************************************************************/
#pragma once

#include <string>



class Detector
{
public:
    void loadData(const char* cfgfile, const char* weightsfile);
    void detectImage(char **names, char *filename, float thresh, int quantized, int dont_show);

private:
    network* m_pNet;
};