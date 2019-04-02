#include <stdio.h>
#include <stdlib.h>

#include "box.h"
#include "pthread.h"

#include "detector.h"
#include "utils/args_util.h"


// get command line parameters and load objects names
void run_detector(int argc, char **argv)
{
    int dont_show = ArgsUtil::find_arg(argc, argv, "-dont_show");
    char *prefix = ArgsUtil::find_char_arg(argc, argv, "-prefix", 0);
    float thresh = ArgsUtil::find_float_arg(argc, argv, "-thresh", .25);
    float iou_thresh = ArgsUtil::find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    char *out_filename = ArgsUtil::find_char_arg(argc, argv, "-out_filename", 0);
    int cam_index = ArgsUtil::find_int_arg(argc, argv, "-c", 0);
    int quantized = ArgsUtil::find_arg(argc, argv, "-quantized");
    int input_calibration = ArgsUtil::find_int_arg(argc, argv, "-input_calibration", 0);
    int frame_skip = ArgsUtil::find_int_arg(argc, argv, "-s", 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [demo/test/] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = 0;                // find_arg(argc, argv, "-clear");

    char *obj_names = argv[3];    // char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;

    // load object names
    char **names = (char**)calloc(10000, sizeof(char *));
    int obj_count = 0;
    FILE* fp;
    char buffer[255];
    fp = fopen(obj_names, "r");
    while (fgets(buffer, 255, (FILE*)fp))
    {
        names[obj_count] = (char*)calloc(strlen(buffer)+1, sizeof(char));
        strcpy(names[obj_count], buffer);
        names[obj_count][strlen(buffer) - 1] = '\0'; //remove newline
        ++obj_count;
    }
    fclose(fp);
    int classes = obj_count;

    if (0 == strcmp(argv[2], "test"))
    {
        //test_detector_cpu(names, cfg, weights, filename, thresh, quantized, dont_show);
        JJ::Detector::instance()->detectImage(names, cfg, weights, filename, thresh, dont_show);
    }


    int i;
    for (i = 0; i < obj_count; ++i) free(names[i]);
    free(names);
}


int main(int argc, char **argv)
{
    run_detector(argc, argv);
    return 0;
}
