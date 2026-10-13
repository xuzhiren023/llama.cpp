#include "llama.h"

#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s INPUT OUTPUT Q3_K_M|Q4_K_M|Q5_K_M\n", argv[0]);
        return 1;
    }
    auto params    = llama_model_quantize_default_params();
    params.nthread = 2;
    if (std::strcmp(argv[3], "Q3_K_M") == 0) {
        params.ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;
    } else if (std::strcmp(argv[3], "Q4_K_M") == 0) {
        params.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    } else if (std::strcmp(argv[3], "Q5_K_M") == 0) {
        params.ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;
    } else {
        return 1;
    }
    return llama_model_quantize(argv[1], argv[2], &params) == 0 ? 0 : 1;
}
