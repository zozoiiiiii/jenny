#include "convolutional_layer.h"

NS_JJ_BEGIN
static inline ACTIVATION get_activation(const std::string& s)
{
    if (s == "logistic") return LOGISTIC;
    if (s == "loggy") return LOGGY;
    if (s == "relu") return RELU;
    if (s ==  "elu") return ELU;
    if (s == "relie") return RELIE;
    if (s == "plse") return PLSE;
    if (s ==  "hardtan") return HARDTAN;
    if (s == "lhtan") return LHTAN;
    if (s == "linear") return LINEAR;
    if (s ==  "ramp") return RAMP;
    if (s ==  "leaky") return LEAKY;
    if (s ==  "tanh") return TANH;
    if (s == "stair") return STAIR;
    //fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}


static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n % 2 == 0) return floor(x / 2.);
    else return (x - n) + floor(x / 2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
static inline float relu_activate(float x) { return x * (x > 0); }
static inline float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
static inline float relie_activate(float x) { return (x > 0) ? x : .01*x; }
static inline float ramp_activate(float x) { return x * (x > 0) + .1*x; }
static inline float leaky_activate(float x) { return (x > 0) ? x : .1*x; }
static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
static inline float plse_activate(float x)
{
    if (x < -4) return .01 * (x + 4);
    if (x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if (x < 0) return .001*x;
    if (x > 1) return .001*(x - 1) + 1;
    return x;
}

static inline ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic") == 0) return LOGISTIC;
    if (strcmp(s, "loggy") == 0) return LOGGY;
    if (strcmp(s, "relu") == 0) return RELU;
    if (strcmp(s, "elu") == 0) return ELU;
    if (strcmp(s, "relie") == 0) return RELIE;
    if (strcmp(s, "plse") == 0) return PLSE;
    if (strcmp(s, "hardtan") == 0) return HARDTAN;
    if (strcmp(s, "lhtan") == 0) return LHTAN;
    if (strcmp(s, "linear") == 0) return LINEAR;
    if (strcmp(s, "ramp") == 0) return RAMP;
    if (strcmp(s, "leaky") == 0) return LEAKY;
    if (strcmp(s, "tanh") == 0) return TANH;
    if (strcmp(s, "stair") == 0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

static float activate(float x, ACTIVATION a)
{
    switch (a) {
    case LINEAR:
        return linear_activate(x);
    case LOGISTIC:
        return logistic_activate(x);
    case LOGGY:
        return loggy_activate(x);
    case RELU:
        return relu_activate(x);
    case ELU:
        return elu_activate(x);
    case RELIE:
        return relie_activate(x);
    case RAMP:
        return ramp_activate(x);
    case LEAKY:
        return leaky_activate(x);
    case TANH:
        return tanh_activate(x);
    case PLSE:
        return plse_activate(x);
    case STAIR:
        return stair_activate(x);
    case HARDTAN:
        return hardtan_activate(x);
    case LHTAN:
        return lhtan_activate(x);
    }
    return 0;
}

void ConvolutionLayer::activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for (i = 0; i < n; ++i) {
        x[i] = activate(x[i], a);
    }
}


bool ConvolutionLayer::load(const IniParser* pParser, int section, size_params params)
{
    int n = pParser->ReadInteger(section, "filters", 1);
    int size = pParser->ReadInteger(section, "size", 1);
    int stride = pParser->ReadInteger(section, "stride", 1);
    int pad = pParser->ReadInteger(section, "pad", 0);
    int padding = pParser->ReadInteger(section, "padding", 0);
    if (pad) padding = size / 2;

    std::string activation_s = pParser->ReadString(section, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c))
        return false;// ("Layer before convolutional layer must output image.");

    int batch_normalize = pParser->ReadInteger(section, "batch_normalize", 0);
    int binary = pParser->ReadInteger(section, "binary", 0);
    int xnor = pParser->ReadInteger(section, "xnor", 0);
    int use_bin_output = pParser->ReadInteger(section, "bin_output", 0);

    int quantized = params.quantized;
    if (params.index == 0 || activation == LINEAR || (params.index > 1 && stride > 1) || size == 1)
        quantized = 0; // disable Quantized for 1st and last layers

    convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, size,
        stride, padding, activation, batch_normalize, binary, xnor, params.net->adam, quantized, use_bin_output);

    layer.flipped = pParser->ReadInteger(section, "flipped", 0);
    layer.dot = pParser->ReadFloat(section, "dot", 0);
    if (params.net->adam)
    {
        layer.B1 = params.net->B1;
        layer.B2 = params.net->B2;
        layer.eps = params.net->eps;
    }

    //return layer;
    return true;
}


float rand_uniform(float min, float max)
{
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand() / RAND_MAX * (max - min)) + min;
}

// utils.c
float rand_scale(float s)
{
    float scale = rand_uniform(1, s);
    if (rand() % 2) return scale;
    return 1. / scale;
}

// utils.c
int rand_int(int min, int max)
{
    if (max < min) {
        int s = min;
        min = max;
        max = s;
    }
    int r = (rand() % (max - min + 1)) + min;
    return r;
}

// utils.c
int constrain_int(int a, int min, int max)
{
    if (a < min) return min;
    if (a > max) return max;
    return a;
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}


size_t get_workspace_size(convolutional_layer l)
{
    if (l.xnor)
        return (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);

    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

layer ConvolutionLayer::make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding,
    ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int quantized, int use_bin_output)
{
    int i;
    
    m_layerInfo.type = CONVOLUTIONAL;
    m_layerInfo.quantized = quantized;

    m_layerInfo.h = h;
    m_layerInfo.w = w;
    m_layerInfo.c = c;
    m_layerInfo.n = n;
    m_layerInfo.binary = binary;
    m_layerInfo.xnor = xnor;
    m_layerInfo.use_bin_output = use_bin_output;
    m_layerInfo.batch = batch;
    m_layerInfo.stride = stride;
    m_layerInfo.size = size;
    m_layerInfo.pad = padding;
    m_layerInfo.batch_normalize = batch_normalize;

    m_weight.weights = (float*)calloc(c*n*size*size, sizeof(float));
    m_layerInfo.weights_int8.assign( c*n*size*size, 0);

    m_weight.biases = (float*)calloc(n, sizeof(float));
    m_layerInfo.biases_quant.assign(n, 0.0f);

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2. / (size*size*c));
    for (i = 0; i < c*n*size*size; ++i)
        m_weight.weights[i] = scale * rand_uniform(-1, 1) ;

    int out_h = convolutional_out_height(m_layerInfo);
    int out_w = convolutional_out_width(m_layerInfo);
    m_layerInfo.out_h = out_h;
    m_layerInfo.out_w = out_w;
    m_layerInfo.out_c = n;
    m_layerInfo.outputs = m_layerInfo.out_h * m_layerInfo.out_w * m_layerInfo.out_c;
    m_layerInfo.inputs = m_layerInfo.w * m_layerInfo.h * m_layerInfo.c;

    m_layerInfo.output = (float*)calloc(m_layerInfo.batch * m_layerInfo.outputs, sizeof(float));
    m_layerInfo.output_int8.assign(m_layerInfo.batch * m_layerInfo.outputs, 0);

    if (binary)
    {
        m_weight.binary_weights = (float*)calloc (c*n*size*size, sizeof(float));
        m_layerInfo.cweights.assign(c*n*size*size, 0);
        m_weight.scales = (float*)calloc(n, sizeof(float));
    }

    if (xnor)
    {
        m_weight.binary_weights = (float*)calloc(c*n*size*size, sizeof(float));
        m_weight.binary_input = (float*)calloc(m_layerInfo.inputs*m_layerInfo.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = m_layerInfo.out_h*m_layerInfo.out_w;
        m_layerInfo.bit_align = src_align + (align - src_align % align);

        m_layerInfo.mean_arr.assign(m_layerInfo.n, 0.0f);
    }

    if (batch_normalize)
    {
        m_weight.scales = (float*)calloc(n, sizeof(float));
        //m_layerInfo.scale_updates = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i)
        {
            m_weight.scales[i] = 1;
        }

        m_layerInfo.mean.assign(n, 0.0f);
        m_layerInfo.variance.assign(n, 0.0f);

        //m_layerInfo.mean_delta = calloc(n, sizeof(float));
        //m_layerInfo.variance_delta = calloc(n, sizeof(float));

        m_weight.rolling_mean = (float*)calloc(n, sizeof(float));
        m_weight.rolling_variance = (float*)calloc(n, sizeof(float));
        m_layerInfo.x.assign(m_layerInfo.batch*m_layerInfo.outputs, 0.0f);
        m_layerInfo.x_norm.assign(m_layerInfo.batch*m_layerInfo.outputs, 0.0f);
    }
    if (adam)
    {
        m_layerInfo.adam = 1;
        m_layerInfo.m.assign( c*n*size*size, 0.0f);
        m_layerInfo.v.assign(c*n*size*size, 0.0f);
    }


    m_layerInfo.workspace_size = get_workspace_size(m_layerInfo);
    m_layerInfo.activation = activation;

    m_layerInfo.bflops = (2.0 * m_layerInfo.n * m_layerInfo.size*m_layerInfo.size*m_layerInfo.c * m_layerInfo.out_h*m_layerInfo.out_w) / 1000000000.;
    if (m_layerInfo.xnor && m_layerInfo.use_bin_output) fprintf(stderr, "convXB");
    else if (m_layerInfo.xnor) fprintf(stderr, "convX ");
    else fprintf(stderr, "conv  ");
    fprintf(stderr, "%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", n, size, size, stride, w, h, c, m_layerInfo.out_w, m_layerInfo.out_h, m_layerInfo.out_c, m_layerInfo.bflops);

    return m_layerInfo;
}









//////////////////////////////////forward//////////////////////////////////////////
void binarize_weights(float *weights, int n, int size, float *binary)
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

void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for (i = 0; i < n; ++i) {
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void activate_array_cpu_custom(float* x, const int n, const ACTIVATION a)
{
    int i = 0;
    if (a == LINEAR) {}
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

void ConvolutionLayer::forward_layer_cpu(network_state state)
{
    layer& l = m_layerInfo;
    int out_h = (l.h + 2 * l.pad - l.size) / l.stride + 1;    // output_height=input_height for stride=1 and pad=1
    int out_w = (l.w + 2 * l.pad - l.size) / l.stride + 1;    // output_width=input_width for stride=1 and pad=1
    int i, f, j;

    // fill zero (ALPHA)
    for (i = 0; i < l.outputs*l.batch; ++i)
        l.output[i] = 0;

    if (l.xnor)
    {
        if (!l.align_bit_weights)
        {
            binarize_weights(m_weight.weights, l.n, l.c*l.size*l.size, m_weight.binary_weights);
            //printf("\n binarize_weights l.align_bit_weights = %p \n", l.align_bit_weights);
        }
        binarize_cpu(state.input, l.c*l.h*l.w*l.batch, m_weight.binary_input);

        m_weight.weights = m_weight.binary_weights;
        state.input = m_weight.binary_input;
    }

    // l.n - number of filters on this layer
    // l.c - channels of input-array
    // l.h - height of input-array
    // l.w - width of input-array
    // l.size - width and height of filters (the same size for all filters)


    // 1. Convolution !!!

    int const out_size = out_h * out_w;

    // 2. Batch normalization
    if (l.batch_normalize) {
        int b;
        for (b = 0; b < l.batch; b++) {
            for (f = 0; f < l.out_c; ++f) {
                for (i = 0; i < out_size; ++i) {
                    int index = f * out_size + i;
                    l.output[index + b * l.outputs] = (l.output[index + b * l.outputs] - m_weight.rolling_mean[f]) / (sqrtf(m_weight.rolling_variance[f]) + .000001f);
                }
            }

            // scale_bias
            for (i = 0; i < l.out_c; ++i) {
                for (j = 0; j < out_size; ++j) {
                    l.output[i*out_size + j + b * l.outputs] *= m_weight.scales[i];
                }
            }
        }
    }

    // 3. Add BIAS
    //if (l.batch_normalize)
    {
        int b;
        for (b = 0; b < l.batch; b++)
        for (i = 0; i < l.n; ++i)
        for (j = 0; j < out_size; ++j)
             l.output[i*out_size + j + b * l.outputs] += m_weight.biases[i];
    }

    // 4. Activation function (LEAKY or LINEAR)
    //if (l.activation == LEAKY) {
    //    for (i = 0; i < l.n*out_size; ++i) {
    //        l.output[i] = leaky_activate(l.output[i]);
    //    }
    //}
    //activate_array_cpu_custom(l.output, l.n*out_size, l.activation);
    activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

}
NS_JJ_END