#include "Header.h"
#include<string>


int main() {
    string dataset_path = "../data/";
    string result_path = "../results/";
    geometric_warping(dataset_path, result_path);
    return 0;
}


void geometric_warping(string dataset_path, string result_path) {
    // read files
    string img_path_hedwig = dataset_path + "hedwig.raw";
    string img_path_raccoon = dataset_path + "raccoon.raw";
    string img_path_bb8 = dataset_path + "bb8.raw";
    int num_rows = 512, num_cols = 512, num_channels = 3;
    unsigned char*** img_hedwig = read_img(img_path_hedwig, num_rows, num_cols, num_channels);
    unsigned char*** img_raccoon = read_img(img_path_raccoon, num_rows, num_cols, num_channels);
    unsigned char*** img_bb8 = read_img(img_path_bb8, num_rows, num_cols, num_channels);
    display_img(img_hedwig, num_rows, num_cols, num_channels, "original_hedwig");
    save_image(img_hedwig, num_rows, num_cols, num_channels, result_path, "original_hedwig");
    display_img(img_raccoon, num_rows, num_cols, num_channels, "original_raccoon");
    save_image(img_raccoon, num_rows, num_cols, num_channels, result_path, "original_raccoon");
    display_img(img_bb8, num_rows, num_cols, num_channels, "original_bb8");
    save_image(img_bb8, num_rows, num_cols, num_channels, result_path, "original_bb8");

    unsigned char*** res_hedwig = create_unsigned_char(num_rows, num_cols, num_channels);
    unsigned char*** res_raccoon = create_unsigned_char(num_rows, num_cols, num_channels);
    unsigned char*** res_bb8 = create_unsigned_char(num_rows, num_cols, num_channels);
    // map each (k, j) into (q', p')
    float r_square = pow(0.5 * num_rows, 2);
    float x, y, u, v, amplify_ratio, k_prime, j_prime;
    int* estimate_values;
    for (int row = 0; row < num_rows; row++) {
        y = num_rows - 0.5 - row;
        y -= num_rows / 2;
        amplify_ratio = 256 / sqrt(r_square - pow(y, 2));
        for (int col = 0; col < num_cols; col++) {
            x = col + 0.5;
            x -= 256;
            u = x * amplify_ratio;
            // map to (k, j)
            k_prime = col + u - x;
            j_prime = row;

            estimate_values = bilinear_interpolation(img_hedwig, num_rows, num_cols, num_channels, k_prime, j_prime);

            if (estimate_values[0] != -1) {
                for (int channel = 0; channel < num_channels; channel++) {
                    res_hedwig[row][col][channel] = estimate_values[channel];
                }
            }

            estimate_values = bilinear_interpolation(img_raccoon, num_rows, num_cols, num_channels, k_prime, j_prime);
            if (estimate_values[0] != -1) {
                for (int channel = 0; channel < num_channels; channel++) {
                    res_raccoon[row][col][channel] = estimate_values[channel];
                }
            }

            estimate_values = bilinear_interpolation(img_bb8, num_rows, num_cols, num_channels, k_prime, j_prime);
            if (estimate_values[0] != -1) {
                for (int channel = 0; channel < num_channels; channel++) {
                    res_bb8[row][col][channel] = estimate_values[channel];
                }
            }
        }
    }
    display_img(res_hedwig, num_rows, num_cols, num_channels, "warp_hedwig");
    save_image(res_hedwig, num_rows, num_cols, num_channels, result_path, "warp_hedwig");
    display_img(res_raccoon, num_rows, num_cols, num_channels, "warp_raccoon");
    save_image(res_raccoon, num_rows, num_cols, num_channels, result_path, "warp_raccoon");
    display_img(res_bb8, num_rows, num_cols, num_channels, "warp_bb8");
    save_image(res_bb8, num_rows, num_cols, num_channels, result_path, "warp_bb8");

    // recover them back to square shape
    unsigned char*** recover_hedwig = create_unsigned_char(num_rows, num_cols, num_channels);
    unsigned char*** recover_raccoon = create_unsigned_char(num_rows, num_cols, num_channels);
    unsigned char*** recover_bb8 = create_unsigned_char(num_rows, num_cols, num_channels);
    float shrink_ratio;
    for (int row = 0; row < num_rows; row++) {
        y = num_rows - 0.5 - row;
        y -= num_rows / 2;
        shrink_ratio = sqrt(r_square - pow(y, 2)) / 256;
        for (int col = 0; col < num_cols; col++) {
            x = col + 0.5;
            x -= 256;
            u = x * shrink_ratio;
            // map to (k, j)
            k_prime = col + u - x;
            j_prime = row;

            estimate_values = special_bilinear_interpolation(res_hedwig, num_rows, num_cols, num_channels, k_prime, j_prime);
            if (estimate_values[0] != -1) {
                for (int channel = 0; channel < num_channels; channel++) {
                    recover_hedwig[row][col][channel] = estimate_values[channel];
                }
            }

            estimate_values = special_bilinear_interpolation(res_raccoon, num_rows, num_cols, num_channels, k_prime, j_prime);
            if (estimate_values[0] != -1) {
                for (int channel = 0; channel < num_channels; channel++) {
                    recover_raccoon[row][col][channel] = estimate_values[channel];
                }
            }

            estimate_values = special_bilinear_interpolation(res_bb8, num_rows, num_cols, num_channels, k_prime, j_prime);
            if (estimate_values[0] != -1) {
                for (int channel = 0; channel < num_channels; channel++) {
                    recover_bb8[row][col][channel] = estimate_values[channel];
                }
            }
        }
    }
    display_img(recover_hedwig, num_rows, num_cols, num_channels, "recover_hedwig");
    save_image(recover_hedwig, num_rows, num_cols, num_channels, result_path, "recover_hedwig");
    display_img(recover_raccoon, num_rows, num_cols, num_channels, "recover_raccoon");
    save_image(recover_raccoon, num_rows, num_cols, num_channels, result_path, "recover_raccoon");
    display_img(recover_bb8, num_rows, num_cols, num_channels, "recover_bb8");
    save_image(recover_bb8, num_rows, num_cols, num_channels, result_path, "recover_bb8");
}
  