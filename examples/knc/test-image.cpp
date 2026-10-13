#include "mtmd-image.h"

#include <cstdio>
#include <stdexcept>

static void check(bool value, const char * name) {
    if (!value) {
        throw std::runtime_error(name);
    }
}

int main() {
    try {
        using processor = mtmd_image_preprocessor_step3vl;
        clip_hparams hp;
        hp.image_size         = 728;
        hp.image_longest_edge = 3024;
        hp.patch_size         = 14;
        hp.n_merge            = 4;
        const auto small      = processor::build_slice_instructions(hp, { 400, 400 });
        check(small.slices.empty(), "small square requires overview only");
        const auto tiled = processor::build_slice_instructions(hp, { 1000, 1000 });
        check(tiled.slices.size() == 4 && tiled.grid_size.width == 2 && tiled.grid_size.height == 2,
              "large square requires four crops");
        const auto below = processor::build_slice_instructions(hp, { 1000, 600 });
        const auto above = processor::build_slice_instructions(hp, { 1000, 630 });
        check(below.slices.size() == 2 && above.slices.size() == 4, "crop rounding threshold");
        const auto wide = processor::build_slice_instructions(hp, { 2400, 300 });
        check(wide.slices.size() == 8 && wide.grid_size.height == 1, "wide image window grid");
        std::puts("PASS: Step3-VL crop geometry");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
