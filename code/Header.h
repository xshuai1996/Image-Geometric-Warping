#ifndef HEAEDER_H
#endif HEAEDER_H


# include <string>
#include <opencv2/opencv.hpp>


using std::string;
using cv::Mat;


void geometric_warping(string dataset_path, string result_path);
unsigned char*** read_img(string img_path, int num_rows, int num_cols, int num_channels);
unsigned char*** create_unsigned_char(int num_rows, int num_cols, int num_channels);
void display_img(unsigned char*** img, int num_rows, int num_cols, int num_channels, string window_name);
void save_image(unsigned char*** image, int num_rows, int num_cols, int num_channels, string path, string name);
Mat unsigned_img_to_mat(unsigned char*** img, int num_rows, int num_cols, int num_channels);
int* bilinear_interpolation(unsigned char*** image, int num_rows, int num_cols, int num_channels, double x, double y);
int* special_bilinear_interpolation(unsigned char*** image, int num_rows, int num_cols, int num_channels, double x, double y);




//
//#include <opencv2/opencv.hpp>
//
//
//using cv::Mat;
//using std::vector;
//
//// problem1.cpp
//void problem_1a(string read_data_folder, string save_result_path);
//
//
//// utils.cpp
//unsigned char*** create_unsigned_char(int num_rows, int num_cols, int num_channels);
//float*** map_to(int num_rows, int num_cols, int space_for_location);
//unsigned char*** read_img(string img_path, int num_rows, int num_cols, int num_channels);
//unsigned char horizontal_interpolation(float left_location, float left_value, float right_location, float right_value, int center);
//void display_img(unsigned char*** img, int num_rows, int num_cols, int num_channels, string window_name);
//double** new_double_matrix(int num_rows, int num_cols);
//vector<vector<double>> select_best_4_pairs(vector<vector<double>> candidates);
//double** inverse_3x3_matrix(double** matrix);
//double** mat_dot(double** mat1, int mat1_rows, int mat1_cols, double** mat2, int mat2_rows, int mat2_cols);
//double** new_position_after_transform(double** transform_matrix, double x, double y);
//void display_double_vector(vector<vector<double>> vec);
//void display_double_array(double** arr, int num_rows, int num_cols);
//int* bilinear_interpolation(unsigned char*** image, int num_rows, int num_cols, int num_channels, double x, double y);
//int* special_bilinear_interpolation(unsigned char*** image, int num_rows, int num_cols, int num_channels, double x, double y);
//unsigned char* calculate_mean_value(unsigned char* position_a, unsigned char* position_b);
//unsigned char*** mark_red_dot(unsigned char*** image, int num_rows, int num_cols, int x, int y);
//vector<vector<int>> construct_STK();
//int check_match_ST(unsigned char*** gray_image, int r, int c);
//int check_match_K(unsigned char*** gray_image, int r, int c);
//unsigned char*** padding_image(unsigned char*** image, int num_rows, int num_cols, int num_channels);
//unsigned char*** STK_process(unsigned char*** image, int num_rows, int num_cols, int num_channels, int type);
//int check_if_in_vector(vector<int> vec, int num);
//unsigned char*** gray_to_01(unsigned char*** image, int num_rows, int num_cols, int num_channels);
//unsigned char*** zeroone_to_255(unsigned char*** image, int num_rows, int num_cols, int num_channels);
//vector<int> DFS(unsigned char*** image, int num_rows, int num_cols);
//void extend_connection(unsigned char*** image, int num_rows, int num_cols, int r, int c, int * area);
//vector<int> BFS(unsigned char*** image, int num_rows, int num_cols);
////Mat opencv_erosion(unsigned char*** image, int num_rows, int num_cols);
//unsigned char*** Mat_to_gray_unsigned_char(Mat mat);
//unsigned char*** erode(unsigned char*** image, int num_rows, int num_cols);
//unsigned char*** copy_unsigned_char(unsigned char*** image, int num_rows, int num_cols, int num_channels);
//unsigned char*** dilation(unsigned char*** image, int num_rows, int num_cols);
//
//
//
//
//
//Mat unsigned_img_to_mat(unsigned char*** img, int num_rows, int num_cols, int num_channels);
//
//void display_img(Mat mat, string window_name);
//unsigned char*** RGB_to_gray(unsigned char*** RGB_img, int num_rows, int num_cols);
//unsigned char*** padding_on_gray(unsigned char*** img, int num_rows, int num_cols, int padding_size);
//float*** filtering_on_gray(float** filter, int filter_size, unsigned char*** img, int num_rows, int num_cols);
//unsigned char*** uniform_on_gary(float*** filtered_img, int num_rows, int num_cols);
//float*** compute_gradient_magnitute(float*** horizontal_gradient, float*** vertical_gradient, int num_rows, int num_cols);
//unsigned char*** gray_to_binary_with_threshold(unsigned char*** img, float threshold, int num_rows, int num_cols);
//float** index_matrix_to_threshold_matrix(int** index_matrix, int size);
//unsigned char*** threshold_matrix_to_result(float** threshold_matrix, int size, unsigned char*** image, int num_rows, int num_cols);
//float*** unsigned_char_image_to_float(unsigned char*** image, int num_rows, int num_cols, int num_channels);
//void save_image(Mat image, string path, string name);
//void save_image(unsigned char*** image, int num_rows, int num_cols, int num_channels, string path, string name);
