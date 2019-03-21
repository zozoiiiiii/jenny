#include "detector.h"
#include "parser/ini_parser.h"
#include "layers/convolutional_layer.h"
#include "layers/yolo_layer.h"
#include "layers/maxpool_layer.h"
#include "layers/route_layer.h"
#include "layers/upsample_layer.h"

#include <time.h>

NS_JJ_BEGIN






// global GPU index: cuda.c
int gpu_index = 0;

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* Detector::get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num)
{
    int selected_num = 0;
    detection_with_class* result_arr = (detection_with_class*)calloc(dets_num, sizeof(detection_with_class));
    int i;
    for (i = 0; i < dets_num; ++i)
    {
        int best_class = -1;
        float best_class_prob = thresh;
        int j;
        for (j = 0; j < dets[i].classes; ++j)
        {
            if (dets[i].prob[j] > best_class_prob) {
                best_class = j;
                best_class_prob = dets[i].prob[j];
            }
        }

        if (best_class >= 0)
        {
            result_arr[selected_num].det = dets[i];
            result_arr[selected_num].best_class = best_class;
            ++selected_num;
        }
    }
    if (selected_detections_num)
        *selected_detections_num = selected_num;
    return result_arr;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr)
{
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w / 2) - (b->det.bbox.x - b->det.bbox.w / 2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr)
{
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

void Detector::draw_detections_v3(ImageInfo im, detection *dets, int num, float thresh, char **names, ImageInfo **alphabet, int classes, int ext_output)
{
    int selected_detections_num;
    detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num);

    // text output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);
    int i;
    for (i = 0; i < selected_detections_num; ++i)
    {
        const int best_class = selected_detections[i].best_class;
        printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
        if (ext_output)
            printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
                round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
                round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
        else
            printf("\n");
        int j;
        for (j = 0; j < classes; ++j) {
            if (selected_detections[i].det.prob[j] > thresh && j != best_class) {
                printf("%s: %.0f%%\n", names[j], selected_detections[i].det.prob[j] * 100);
            }
        }
    }

    // ImageInfo output
    qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);
    for (i = 0; i < selected_detections_num; ++i) {
        int width = im.h * .006;
        if (width < 1)
            width = 1;

        /*
        if(0){
        width = pow(prob, 1./2.)*10+1;
        alphabet = 0;
        }
        */

        //printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
        int offset = selected_detections[i].best_class * 123457 % classes;
        float red = ImageUtil::get_color(2, offset, classes);
        float green = ImageUtil::get_color(1, offset, classes);
        float blue = ImageUtil::get_color(0, offset, classes);
        float rgb[3];

        //width = prob*20+2;

        rgb[0] = red;
        rgb[1] = green;
        rgb[2] = blue;
        box b = selected_detections[i].det.bbox;
        //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

        int left = (b.x - b.w / 2.)*im.w;
        int right = (b.x + b.w / 2.)*im.w;
        int top = (b.y - b.h / 2.)*im.h;
        int bot = (b.y + b.h / 2.)*im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;

        ImageUtil::draw_box_width(im, left, top, right, bot, width, red, green, blue);
    }
    free(selected_detections);
}


// fuse convolutional and batch_norm weights into one convolutional-layer
void Detector::yolov2_fuse_conv_batchnorm(network* net)
{
  
}

void Detector::calculate_binary_weights(network* net)
{
   

}

void binarize_weights(std::vector<float> weights, int n, int size, std::vector<float>& binary)
{
    int i, f;
    for (f = 0; f < n; ++f) {
        float mean = 0;
        for (i = 0; i < size; ++i) {
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for (i = 0; i < size; ++i) {
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}


void float_to_bit(std::vector<float> src, std::vector<unsigned char>& dst, size_t size)
{
    size_t dst_size = size / 8 + 1;
    //memset(dst, 0, dst_size);

    size_t i;
    char *byte_arr = (char*)calloc(size, sizeof(char));
    for (i = 0; i < size; ++i) {
        if (src[i] > 0) byte_arr[i] = 1;
    }

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for (i = 0; i < size; i += 8) {
        char dst_tmp = 0;
        dst_tmp |= byte_arr[i + 0] << 0;
        dst_tmp |= byte_arr[i + 1] << 1;
        dst_tmp |= byte_arr[i + 2] << 2;
        dst_tmp |= byte_arr[i + 3] << 3;
        dst_tmp |= byte_arr[i + 4] << 4;
        dst_tmp |= byte_arr[i + 5] << 5;
        dst_tmp |= byte_arr[i + 6] << 6;
        dst_tmp |= byte_arr[i + 7] << 7;
        dst[i / 8] = dst_tmp;
    }
    free(byte_arr);
}

void get_mean_array(float *src, size_t size, size_t filters, float *mean_arr) {
    size_t i, counter;
    counter = 0;
    for (i = 0; i < size; i += size / filters) {
        mean_arr[counter++] = fabs(src[i]);
    }
}

void Detector::binary_align_weights(layer *l)
{
   
}






Detector* Detector::instance()
{
    static Detector s_detector;
    return &s_detector;
}

void Detector::detectImage(char **names, char *cfgfile, char *weightfile, char *filename, float thresh, int quantized, int dont_show)
{
    if (!filename)
        return;

    //ImageInfo **alphabet = load_alphabet();            // image.c
    ImageInfo **alphabet = NULL;

    // 1. read config file, like convolution layer
    network* net = readConfigFile(cfgfile, 1, quantized);    // parser.c
    if (weightfile)
    {
        // 2. read weight file, init the layer information in the network. cutoff == net.n, means do not cut off any layer
        readWeightFile(net, weightfile, net->n);
    }




    //set_batch_network(&net, 1);                    // network.c
    srand(2222222);
    yolov2_fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    // cancel here
    if (quantized)
    {
        printf("\n\n Quantinization! \n\n");
        //quantinization_and_get_multipliers(net);
    }

    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms = .4;
    while (1)
    {
        // 3. open image
        strncpy(input, filename, 256);
        ImageInfo im = ImageUtil::load_image(input, 0, 0, 3);            // image.c
        ImageInfo sized = ImageUtil::resize_image(im, net->w, net->h);    // image.c
        layer* l = net->jjLayers[net->n - 1]->getLayer();


        float *X = sized.data;
        time = clock();
        //network_predict(net, X);

        // 4. predict, the key
        //network_predict_cpu(net, X);

        printf("%s: Predicted in %f seconds.\n", input, (float)(clock() - time) / CLOCKS_PER_SEC); //sec(clock() - time));
        //get_region_boxes_cpu(l, 1, 1, thresh, probs, boxes, 0, 0);            // get_region_boxes(): region_layer.c

        // 5. save to ImageInfo or show directly
        float hier_thresh = 0.5;
        int ext_output = 1, letterbox = 0, nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
        if (nms)
            do_nms_sort(dets, nboxes, l->classes, nms);

        //draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
        ImageUtil::save_image_png(im, "predictions");    // image.c
        ImageUtil::free_image(im);                    // image.c
        ImageUtil::free_image(sized);                // image.c
        if (filename)
            break;
    }

}

void Detector::fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int j;
//     for (j = 0; j < net->n; ++j)
//     {
//         layer l = net->layers[j];
//         if (l.type == YOLO)
//         {
//             int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
//             dets += count;
//         }
// 
//         if (l.type == REGION)
//         {
//             custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
//             dets += l.w*l.h*l.n;
//         }
//     }
}

detection * Detector::get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
//     detection *dets = make_network_boxes(net, thresh, num);
//     fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
//     return dets;
    return nullptr;
}


void load_convolutional_weights_cpu(ConvolutionLayer* pLayer, FILE *fp)
{
    layer* pLayerInfo = pLayer->getLayer();
    ConvolutionWeight* pWeight = pLayer->getWeight();
    int num = pLayerInfo->n * pLayerInfo->c * pLayerInfo->size * pLayerInfo->size;
    int n = pLayerInfo->n;
    fread(pWeight->biases, sizeof(float), n, fp);
    if (pLayerInfo->batch_normalize && (!pLayerInfo->dontloadscales))
    {
        fread(pWeight->scales, sizeof(float), n, fp);
        fread(pWeight->rolling_mean, sizeof(float), n, fp);
        fread(pWeight->rolling_variance, sizeof(float), n, fp);
    }
    fread(pWeight->weights, sizeof(float), num, fp);
}

bool Detector::readWeightFile(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if (!fp)
        return false;

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2)
    {
        fread(net->seen, sizeof(uint64_t), 1, fp);
    }
    else
    {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    //int transpose = (major > 1000) || (minor > 1000);

    int i;
    for (i = 0; i < net->n && i < cutoff; ++i)
    {
        ILayer* pLayer = net->jjLayers[i];
        if (pLayer->getLayer()->dontload)
            continue;

        if (pLayer->getLayer()->type == CONVOLUTIONAL)
        {
            load_convolutional_weights_cpu((ConvolutionLayer*)pLayer, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
    return true;
}


JJ::network* Detector::readConfigFile(const char* filename, int batch, int quantized)
{
    IniParser parser(filename);
    
    JJ::network* pNetWork = new JJ::network;
    pNetWork->n = parser.GetSectionCount() - 1; //  layer count
    pNetWork->quantized = quantized;
    pNetWork->do_input_calibration = 0;
    

    size_params params;
    params.quantized = quantized;

    if (!parseNetOptions(&parser, pNetWork))
        return nullptr;

    params.h = pNetWork->h;
    params.w = pNetWork->w;
    params.c = pNetWork->c;
    params.inputs = pNetWork->inputs;
    if (batch > 0)
        pNetWork->batch = batch;
    params.batch = pNetWork->batch;
    params.time_steps = pNetWork->time_steps;
    params.net = pNetWork;



    size_t workspace_size = 0;    

    // read convolutional, maxpool
    for(int i=0; i<pNetWork->n; i++)
    {
        int sectionIndex = i + 1;
        params.index = i;
        fprintf(stderr, "%5d ", i);
        std::string sectionName = parser.GetSectionByIndex(sectionIndex);
        int itemCnt = parser.GetSectionItemCount(sectionIndex);

        //s = (section *)n->val;
        //options = s->options;

        // parse data into this layer and save to net
        //JJ::layer l;
        JJ::ILayer* pLayer = nullptr;
        JJ::LAYER_TYPE lt = string_to_layer_type(sectionName.c_str());

        // [convolutional]
        if (lt == JJ::CONVOLUTIONAL)
        {
            //l = parse_convolutional(options, params);
            pLayer = new ConvolutionLayer;
            pLayer->load(&parser, sectionIndex, params);
        }
        //else if (lt == REGION) {
            //l = parse_region(options, params);
        //}
        else if (lt == YOLO) {
            //l = parse_yolo(options, params);
            pLayer = new YoloLayer;
            pLayer->load(&parser, sectionIndex, params);
        }
        //else if (lt == SOFTMAX) {
            //l = parse_softmax(options, params);
            //net.hierarchy = l.softmax_tree;
        //}
        else if (lt == MAXPOOL) {
            //l = parse_maxpool(options, params);
            pLayer = new MaxpoolLayer;
            pLayer->load(&parser, sectionIndex, params);
        }
        //else if (lt == REORG) {
            //l = parse_reorg(options, params);
        //}
        else if (lt == ROUTE) {
            //l = parse_route(options, params, net);
            pLayer = new RouteLayer;
            pLayer->load(&parser, sectionIndex, params);
        }
        else if (lt == UPSAMPLE) {
            //l = parse_upsample(options, params, net);
            pLayer = new UpsampleLayer;
            pLayer->load(&parser, sectionIndex, params);
        }
        //else if (lt == SHORTCUT) {
            //l = parse_shortcut(options, params, net);
        //}
        else {
            fprintf(stderr, "Type not recognized: %s\n", sectionName.c_str());
        }

        layer* pLayerInfo = pLayer->getLayer();
        pLayerInfo->dontload = parser.ReadInteger(sectionIndex, "dontload", 0);
        pLayerInfo->dontloadscales = parser.ReadInteger(sectionIndex, "dontloadscales", 0);
        //option_unused(options);

        // save this layer
        pNetWork->jjLayers.push_back(pLayer);
        if (pLayerInfo->workspace_size > workspace_size)
            workspace_size = pLayerInfo->workspace_size;

        //free_section(s);
        //n = n->next;
        //++count;
        //if (n)
        {
            params.h = pLayerInfo->out_h;
            params.w = pLayerInfo->out_w;
            params.c = pLayerInfo->out_c;
            params.inputs = pLayerInfo->outputs;
        }
    }


    //free_list(sections);
    pNetWork->outputs = NetWork::get_network_output_size(pNetWork);
    pNetWork->output = NetWork::get_network_output(pNetWork);
    if (workspace_size)
    {
        //printf("%ld\n", workspace_size);
        pNetWork->workspace.assign(workspace_size, 0.0f);
    }
    return pNetWork;
}




JJ::learning_rate_policy Detector::get_policy(const char *s)
{
    if (strcmp(s, "random") == 0) return JJ::RANDOM;
    if (strcmp(s, "poly") == 0) return JJ::POLY;
    if (strcmp(s, "constant") == 0) return JJ::CONSTANT;
    if (strcmp(s, "step") == 0) return JJ::STEP;
    if (strcmp(s, "exp") == 0) return JJ::EXP;
    if (strcmp(s, "sigmoid") == 0) return JJ::SIG;
    if (strcmp(s, "steps") == 0) return JJ::STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return JJ::CONSTANT;
}

JJ::LAYER_TYPE Detector::string_to_layer_type(const std::string& type)
{
    if (type == "yolo") return JJ::YOLO;
    if (type == "region") return JJ::REGION;
    if (type ==  "conv" || type == "convolutional") return JJ::CONVOLUTIONAL;
    if (type == "net"  || type == "network") return JJ::NETWORK;
    if (type, "max"|| type == "maxpool") return JJ::MAXPOOL;
    if (type, "reorg") return JJ::REORG;
    if (type, "upsample") return JJ::UPSAMPLE;
    if (type, "shortcut") return JJ::SHORTCUT;
    if (type, "soft" || type == "softmax") return JJ::SOFTMAX;
    if (type, "route") return JJ::ROUTE;
    return JJ::BLANK;
}

bool Detector::parseNetOptions(const IniParser* pIniParser, JJ::network* net)
{
    int netSectionIndex = 0;
    net->batch = pIniParser->ReadInteger(netSectionIndex, "batch", 1);
    net->learning_rate = pIniParser->ReadFloat(netSectionIndex, "learning_rate", .001);
    net->momentum = pIniParser->ReadFloat(netSectionIndex, "momentum", .9);
    net->decay = pIniParser->ReadFloat(netSectionIndex, "decay", .0001);
    int subdivs = pIniParser->ReadInteger(netSectionIndex, "subdivisions", 1);
    net->time_steps = pIniParser->ReadInteger(netSectionIndex, "time_steps", 1);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;

    std::string input_calibration= pIniParser->ReadString(netSectionIndex, "input_calibration", 0);
    if (!input_calibration.empty())
    {
        StringUtil::splitFloat(net->input_calibration, input_calibration, ",");
    }

    net->adam = pIniParser->ReadInteger(netSectionIndex, "adam", 0);
    if (net->adam)
    {
        net->B1 = pIniParser->ReadFloat(netSectionIndex, "B1", .9);
        net->B2 = pIniParser->ReadFloat(netSectionIndex, "B2", .999);
        net->eps = pIniParser->ReadFloat(netSectionIndex, "eps", .000001);
    }

    net->h = pIniParser->ReadInteger(netSectionIndex, "height", 0);
    net->w = pIniParser->ReadInteger(netSectionIndex, "width", 0);
    net->c = pIniParser->ReadInteger(netSectionIndex, "channels", 0);
    net->inputs = pIniParser->ReadInteger(netSectionIndex, "inputs", net->h * net->w * net->c);
    net->max_crop = pIniParser->ReadInteger(netSectionIndex, "max_crop", net->w * 2);
    net->min_crop = pIniParser->ReadInteger(netSectionIndex, "min_crop", net->w);

    net->angle = pIniParser->ReadFloat(netSectionIndex, "angle", 0);
    net->aspect = pIniParser->ReadFloat(netSectionIndex, "aspect", 1);
    net->saturation = pIniParser->ReadFloat(netSectionIndex, "saturation", 1);
    net->exposure = pIniParser->ReadFloat(netSectionIndex, "exposure", 1);
    net->hue = pIniParser->ReadFloat(netSectionIndex, "hue", 0);

    if (!net->inputs && !(net->h && net->w && net->c))
        return false;
        //error("No input parameters supplied");

    std::string policy_s = pIniParser->ReadString(netSectionIndex, "policy", "constant");
    net->policy = get_policy(policy_s.c_str());
    net->burn_in = pIniParser->ReadInteger(netSectionIndex, "burn_in", 0);
    if (net->policy == JJ::STEP)
    {
        net->step = pIniParser->ReadInteger(netSectionIndex, "step", 1);
        net->scale = pIniParser->ReadFloat(netSectionIndex, "scale", 1);
    }
    else if (net->policy == JJ::STEPS)
    {
        std::string l = pIniParser->ReadString(netSectionIndex, "steps");
        std::string p = pIniParser->ReadString(netSectionIndex, "scales");
        if (l.empty() || p.empty())
            return false;// ("STEPS policy must have steps and scales in cfg file");

        StringUtil::splitInt(net->steps, l, ",");
        StringUtil::splitFloat(net->scales, p, ",");

        //net->num_steps = net->steps.size();
    }
    else if (net->policy == JJ::EXP)
    {
        net->gamma = pIniParser->ReadFloat(netSectionIndex, "gamma", 1);
    }
    else if (net->policy == JJ::SIG)
    {
        net->gamma = pIniParser->ReadFloat(netSectionIndex, "gamma", 1);
        net->step = pIniParser->ReadInteger(netSectionIndex, "step", 1);
    }
    else if (net->policy == JJ::POLY || net->policy == JJ::RANDOM)
    {
        net->power = pIniParser->ReadFloat(netSectionIndex, "power", 1);
    }
    net->max_batches = pIniParser->ReadInteger(netSectionIndex, "max_batches", 0);
    return true;
}




void Detector::yolov2_forward_network_cpu(network net, network_state state)
{
    //state.workspace = net.workspace;
    int i;
    for (i = 0; i < net.n; ++i)
    {
        state.index = i;
        ILayer* pLayer = net.jjLayers[i];
        //pLayer->forward_layer_cpu(pLayer->getLayer(), state);

        //state.input = l.output;
    }
}


// detect on CPU
float * Detector::network_predict_cpu(network net, float *input)
{
    return nullptr;
// 
//     network_state state;
//     state.net = net;
//     state.index = 0;
//     state.input = input;
//     state.truth = 0;
//     state.train = 0;
//     state.delta = 0;
//     yolov2_forward_network_cpu(net, state);    // network on CPU
//                                             //float *out = get_network_output(net);
// 
//     // updated by junliang, begin, do not care about return 
//     int i;
//     for (i = net.n - 1; i > 0; --i) if (net.layers[i].type != COST) break;
//     return net.layers[i].output;
}

NS_JJ_END