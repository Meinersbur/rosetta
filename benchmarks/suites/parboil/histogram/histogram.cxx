// IGNORE: add_benchmark(serial)

#include <rosetta.h>

static void kernel(pbsize_t img_width, pbsize_t img_height, unsigned* img, uint8_t* histo) {
    for (pbsize_t i = 0; i < img_width*img_height; ++i) {
      const unsigned int value = img[i];
      if (histo[value] < UINT8_MAX) {
        ++histo[value];
      }
  }
}

void run(State &state, pbsize_t n) {
  pbsize_t img_width = n;
  pbsize_t img_height = n - n/3;
  pbsize_t hist_width = 255;
  pbsize_t hist_height = 3;

  auto img = state.allocate_array<unsigned>({img_width*img_height}, /*fakedata*/ true, /*verify*/ false, "img");
  auto histo = state.allocate_array<uint8_t>({hist_width*hist_height}, /*fakedata*/ false, /*verify*/ true, "histo");

  for (auto &&_ : state)
    kernel(img_width, img_height, img.data(), histo.data());
}
