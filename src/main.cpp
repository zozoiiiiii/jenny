#include <stdio.h>
#include <stdlib.h>

#include "box.h"
#include "pthread.h"

#include "detector.h"
#include "utils/args_util.h"


//yolo_cpu.exe detector [train/test/valid] [object names] [cfg] [weights(optional)] [filename(optional)]
void run_detector(int argc, char **argv)
{
    float thresh = ArgsUtil::find_float_arg(argc, argv, "-thresh", .25);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int clear = 0;                // find_arg(argc, argv, "-clear");

    char *obj_names = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;

    // load object names
    std::vector<std::string> names;
    int obj_count = 0;
    FILE* fp;
    char buffer[255];
    fp = fopen(obj_names, "r");
    while (fgets(buffer, 255, (FILE*)fp))
    {
        buffer[strlen(buffer) - 1] = '\0';  // remove newline
        names.push_back(buffer);
    }
    fclose(fp);

    if (0 == strcmp(argv[2], "test"))
    {
        //test_detector_cpu(names, cfg, weights, filename, thresh, quantized, dont_show);
        JJ::Detector::instance()->test(names, cfg, weights, filename, thresh);
    }
}


int main(int argc, char **argv)
{
    run_detector(argc, argv);
    return 0;
}
