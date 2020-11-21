#include "Header.h"
#include <iostream>


using std::cout;
using std::endl;
using cv::waitKey;
using cv::destroyAllWindows;


unsigned char*** read_img(string img_path, int num_rows, int num_cols, int num_channels) {
	unsigned char*** img = create_unsigned_char(num_rows, num_cols, num_channels);
	// read img from file
	FILE* file;
	errno_t file_err = fopen_s(&file, (img_path).c_str(), "rb");
	if (file_err != 0) {
		cout << "Error: Fail to read file \"" + img_path + "\". An unassigned array will be returned." << endl;
	}
	else {
		int product_size = num_rows * num_cols * num_channels;
		unsigned char * flatten_img = new unsigned char[product_size];
		fread(flatten_img, sizeof(unsigned char), product_size, file);
		fclose(file);
		for (int row = 0; row < num_rows; row++) {
			for (int col = 0; col < num_cols; col++) {
				for (int channel = 0; channel < num_channels; channel++) {
					img[row][col][channel] = flatten_img[row * num_cols * num_channels + col * num_channels + channel];
				}
			}
		}
	}
	return img;
}


unsigned char*** create_unsigned_char(int num_rows, int num_cols, int num_channels) {
	// allocate new space
	unsigned char*** space = new unsigned char** [num_rows];
	for (int row = 0; row < num_rows; row++) {
		space[row] = new unsigned char* [num_cols];
		for (int col = 0; col < num_cols; col++) {
			space[row][col] = new unsigned char[num_channels];
		}
	}
	// initialize to 0
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			for (int channel = 0; channel < num_channels; channel++) {
				space[row][col][channel] = 0;
			}
		}
	}
	return space;
}


void display_img(unsigned char*** img, int num_rows, int num_cols, int num_channels, string window_name) {
	Mat mat = unsigned_img_to_mat(img, num_rows, num_cols, num_channels);
	imshow(window_name, mat);
	waitKey(0);
	destroyAllWindows();
}


void save_image(unsigned char*** image, int num_rows, int num_cols, int num_channels, string path, string name) {
	Mat mat = unsigned_img_to_mat(image, num_rows, num_cols, num_channels);
	imwrite(path + name + ".jpg", mat);
}


Mat unsigned_img_to_mat(unsigned char*** img, int num_rows, int num_cols, int num_channels) {
	// reshape img to 1d array so that it can be Mat.data
	int product_size = num_rows * num_cols * num_channels;
	unsigned char* reshaped_img = new unsigned char[product_size];
	for (int row = 0; row < num_rows; row++) {
		for (int col = 0; col < num_cols; col++) {
			if (num_channels == 3) {	// color image
				// in opencv the rank is BGR
				reshaped_img[row * num_cols * num_channels + col * num_channels + 0] = img[row][col][2];
				reshaped_img[row * num_cols * num_channels + col * num_channels + 1] = img[row][col][1];
				reshaped_img[row * num_cols * num_channels + col * num_channels + 2] = img[row][col][0];
			}
			else {	// gray image
				reshaped_img[row * num_cols * num_channels + col * num_channels] = img[row][col][0];
			}
		}
	}
	Mat mat;
	if (num_channels == 3) {
		mat = Mat(num_rows, num_cols, CV_8UC3);	// CV_8U: 8-bit unsigned integer; C3: 3 channels
	}
	else {
		mat = Mat(num_rows, num_cols, CV_8UC1);
	}
	mat.data = reshaped_img;
	return mat;
}


int* bilinear_interpolation(unsigned char*** image, int num_rows, int num_cols, int num_channels, double x, double y) {
	int* ret = new int[num_channels];
	// if out of boundary
	if (x<=-1 or x>=num_cols or y<=-1 or y>=num_rows) {
		for (int k = 0; k < num_channels; k++) {
			ret[k] = -1;
		}
		return ret;
	}
	int x1 = (int)x;
	int x2 = (int)x + 1;
	int y1 = (int)y + 1;
	int y2 = (int)y;
	int value_x1 = (x < 0) ? x1 + 1 : x1;
	int value_x2 = (x >= num_cols - 1) ? x2 - 1 : x2;
	int value_y1 = (y >= num_rows - 1) ? y1 - 1 : y1;
	int value_y2 = (y < 0) ? y2 + 1 : y2;
	double Q11, Q12, Q21, Q22;
	double calculate_value;
	for (int k = 0; k < num_channels; k++) {
		Q11 = image[value_y1][value_x1][k];
		Q12 = image[value_y2][value_x1][k];
		Q21 = image[value_y1][value_x2][k];
		Q22 = image[value_y2][value_x2][k];
		calculate_value = Q11 * (x2 - x) * (y - y2) + Q21 * (x - x1) * (y - y2) + Q12 * (x2 - x) * (y1 - y) + Q22 * (x - x1) * (y1 - y);

		if (calculate_value <= 0) {
			ret[k] = 0;
		}
		else if (calculate_value >= 255) {
			ret[k] = 255;
		}
		else {
			ret[k] = (int)calculate_value;
		}
	}
	return ret;
}


int* special_bilinear_interpolation(unsigned char*** image, int num_rows, int num_cols, int num_channels, double x, double y) {
	int* ret = new int[num_channels];
	// if out of boundary
	if (x<-1 or x>num_cols or y<-1 or y>num_rows) {
		for (int k = 0; k < num_channels; k++) {
			ret[k] = -1;
		}
		return ret;
	}
	int x1 = (int)x;
	int x2 = (int)x + 1;
	int y1 = (int)y + 1;
	int y2 = (int)y;
	int value_x1 = (x < 0) ? x1 + 1 : x1;
	int value_x2 = (x >= num_cols - 1) ? x2 - 1 : x2;
	int value_y1 = (y >= num_rows - 1) ? y1 - 1 : y1;
	int value_y2 = (y < 0) ? y2 + 1 : y2;
	double special_x1 = value_x1 + 0.5;
	double special_x2 = value_x2 + 0.5;
	double special_y1 = value_y1 + 0.5;
	double special_y2 = value_y2 + 0.5;
	if ((pow(special_x1 - 256, 2) + pow(special_y1 - 256, 2) >= pow(256, 2))
		|| (pow(special_x1 - 256, 2) + pow(special_y2 - 256, 2) >= pow(256, 2))) {
		value_x1 += 1;
		if (y > 255) {
			value_y1 -= 1;
			value_y2 -= 1;
		}
	}
	if ((pow(special_x2 - 256, 2) + pow(special_y1 - 256, 2) >= pow(256, 2))
		|| (pow(special_x2 - 256, 2) + pow(special_y2 - 256, 2) >= pow(256, 2))) {
		value_x2 -= 1;		
		if (y > 255) {
			value_y1 -= 1;
			value_y2 -= 1;
		}
	}

	double Q11, Q12, Q21, Q22;
	double calculate_value;
	for (int k = 0; k < num_channels; k++) {
		Q11 = image[value_y1][value_x1][k];
		Q12 = image[value_y2][value_x1][k];
		Q21 = image[value_y1][value_x2][k];
		Q22 = image[value_y2][value_x2][k];
		calculate_value = Q11 * (x2 - x) * (y - y2) + Q21 * (x - x1) * (y - y2) + Q12 * (x2 - x) * (y1 - y) + Q22 * (x - x1) * (y1 - y);
		if (calculate_value <= 0) {
			ret[k] = 0;
		}
		else if (calculate_value >= 255) {
			ret[k] = 255;
		}
		else {
			ret[k] = (int)calculate_value;
		}
	}
	return ret;
}




//#include <string>
//#include <math.h>
//
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include<opencv2/highgui.hpp>
////#include <highgui.h>
//

//using std::string;
//using cv::imshow;
//

//using cv::imwrite;
//using std::vector;
//using std::cin;
//using std::queue;
//using std::pair;
//using cv::getStructuringElement;
//using cv::MORPH_RECT;
//using cv::Size;
//using cv::Point;
//using cv::erode;
//

//
//float*** map_to(int num_rows, int num_cols, int space_for_location) {
//	// allocate new space
//	float*** matrix = new float** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		matrix[row] = new float* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			matrix[row][col] = new float[space_for_location];
//		}
//	}
//	// initialize to 0
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			for (int space = 0; space < space_for_location; space++) {
//				matrix[row][col][space] = 0.0;
//			}
//		}
//	}
//	return matrix;
//}
//

//
//unsigned char horizontal_interpolation(float left_location, float left_value, float right_location, float right_value, int center) {
//	float left_distance = center - left_location;
//	float right_distance = right_location - center;
//	float float_res = left_value + (right_value - left_value) * left_distance / (left_distance + right_distance);
//	unsigned char res = float_res;
//	if (float_res < 0) {
//		res = 0;
//	}
//	if (float_res > 255) {
//		res = 255;
//	}
//	return res;
//}
//

//

//
//void display_img(Mat mat, string window_name) {
//	imshow(window_name, mat);
//	waitKey(0);
//	destroyAllWindows();
//}
//

//
//void save_image(Mat image, string path, string name) {
//	imwrite(path + name + ".jpg", image);
//}
//
//unsigned char*** RGB_to_gray(unsigned char*** RGB_img, int num_rows, int num_cols) {
//	// allocate new space
//	unsigned char*** gray_img = create_unsigned_char(num_rows, num_cols, 1);
//	// Y = 0.2989 x R + 0.5870 x G + 0.1140 x B
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			float R = RGB_img[row][col][0], G = RGB_img[row][col][1], B = RGB_img[row][col][2];
//			float weighted_sum = 0.2989 * R + 0.5870 * G + 0.1140 * B;
//			gray_img[row][col][0] = weighted_sum;
//		}
//	}
//	return gray_img;
//}
//
//vector<vector<double>> select_best_4_pairs(vector<vector<double>> candidates) {
//	/* simply fond the upper most, b*/
//	double bottom = -1000, top = 1000, left = 1000, right = -1000;
//	vector<double> top_vector, bottom_vector, left_vector, right_vector;
//	for (int i = 0; i < candidates.size(); i++) {
//		if (candidates[i][1] > bottom) {
//			bottom = candidates[i][1];
//			bottom_vector = candidates[i];
//		}
//		if (candidates[i][1] < top) {
//			top = candidates[i][1];
//			top_vector = candidates[i];
//		}
//		if (candidates[i][0] < left) {
//			left = candidates[i][0];
//			left_vector = candidates[i];
//		}
//		if (candidates[i][0] > right) {
//			right = candidates[i][0];
//			right_vector = candidates[i];
//		}
//	}
//	vector<vector<double>> ret{ top_vector, bottom_vector, left_vector, right_vector };
//	return ret;
//}
//
//double** new_double_matrix(int num_rows, int num_cols) {
//	double** ret = new double* [num_rows];
//	for (int i = 0; i < num_rows; i++) {
//		ret[i] = new double[num_cols];
//	}
//	return ret;
//}
//
//double** inverse_3x3_matrix(double** matrix) {
//	double** ret = new_double_matrix(3, 3);
//	double determinant = 0;
//
//	//finding determinant
//	for (int i = 0; i < 3; i++) {
//		determinant += (matrix[0][i] * (matrix[1][(i + 1) % 3] * matrix[2][(i + 2) % 3] - matrix[1][(i + 2) % 3] * matrix[2][(i + 1) % 3]));
//	}
//	for (int i = 0; i < 3; i++) {
//		for (int j = 0; j < 3; j++) {
//			ret[i][j] = ((matrix[(j + 1) % 3][(i + 1) % 3] * matrix[(j + 2) % 3][(i + 2) % 3]) - 
//				(matrix[(j + 1) % 3][(i + 2) % 3] * matrix[(j + 2) % 3][(i + 1) % 3])) / determinant;
//		}
//	}
//	return ret;
//}
//
//double** mat_dot(double** mat1, int mat1_rows, int mat1_cols, double** mat2, int mat2_rows, int mat2_cols) {
//	double** ret = new_double_matrix(mat1_rows, mat2_cols);
//	double element;
//	for (int i = 0; i < mat1_rows; i++) {
//		for (int j = 0; j < mat2_cols; j++) {
//			element = 0;
//			for (int k = 0; k < mat1_cols; k++) {
//				element += mat1[i][k] * mat2[k][j];
//				cout << element << " ";
//			}
//			ret[i][j] = element;
//		}
//	}
//	return ret;
//}
//
//double** new_position_after_transform(double** transform_matrix, double x, double y) {
//	double** old_position = new_double_matrix(3, 1);
//	old_position[0][0] = x; old_position[1][0] = y; old_position[2][0] = 1;
//	double** new_position = mat_dot(transform_matrix, 3, 3, old_position, 3, 1);
//
//	cout << new_position[0][0] << " " << new_position[1][0] << " " << new_position[2][0] << endl;
//
//	new_position[0][0] = new_position[0][0] / new_position[2][0];
//	new_position[1][0] = new_position[1][0] / new_position[2][0];
//	new_position[2][0] = new_position[2][0] / new_position[2][0];
//	return new_position;
//}
//
//void display_double_vector(vector<vector<double>> vec) {
//	for (int i = 0; i < vec.size(); i++) {
//		for (int j = 0; j < vec[0].size(); j++) {
//			cout << vec[i][j] << " ";
//		}
//		cout << endl;
//	}
//	cout << endl;
//}
//
//void display_double_array(double** arr, int num_rows, int num_cols) {
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			cout << arr[i][j] << " ";
//		}
//		cout << endl;
//	}
//	cout << endl;
//}
//
//unsigned char*** padding_on_gray(unsigned char*** img, int num_rows, int num_cols, int padding_size) {
//	unsigned char*** padding_img = new unsigned char** [num_rows + 2 * padding_size];
//	for (int row = 0; row < num_rows + 2 * padding_size; row++) {
//		padding_img[row] = new unsigned char* [num_cols + 2 * padding_size];
//		for (int col = 0; col < num_cols + 2 * padding_size; col++) {
//			padding_img[row][col] = new unsigned char[1];
//		}
//	}
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			padding_img[row + padding_size][col + padding_size][0] = img[row][col][0];
//		}
//	}
//	for (int i = 0; i < padding_size; i++) {
//		for (int j = padding_size; j < num_rows + padding_size; j++) {
//			padding_img[j][i][0] = padding_img[j][padding_size][0];
//			padding_img[j][num_rows + 2 * padding_size - 1 - i][0] = padding_img[j][num_rows + padding_size - 1][0];
//		}
//	}
//	for (int i = 0; i < padding_size; i++) {
//		for (int j = 0; j < num_cols + 2 * padding_size; j++) {
//			padding_img[i][j][0] = padding_img[padding_size][j][0];
//			padding_img[num_rows + 2 * padding_size - 1 - i][j][0] = padding_img[num_rows + padding_size - 1][j][0];
//		}
//	}
//	return padding_img;
//}
//
//float*** filtering_on_gray(float** filter, int filter_size, unsigned char*** img, int num_rows, int num_cols) {
//	int padding_size = (filter_size - 1) / 2;
//	unsigned char*** padding_img = padding_on_gray(img, num_rows, num_cols, padding_size);
//	float*** filtered_img = new float** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		filtered_img[row] = new float* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			filtered_img[row][col] = new float[1];
//		}
//	}
//	float add_sum, img_element, filter_element;
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			add_sum = 0;
//			for (int i = 0; i < filter_size; i++) {
//				for (int j = 0; j < filter_size; j++) {
//					img_element = padding_img[row + i][col + j][0];
//					filter_element = filter[i][j];
//					add_sum += img_element * filter_element;
//				}
//			}
//			filtered_img[row][col][0] = add_sum;
//		}
//	}
//	return filtered_img;
//}
//
//unsigned char*** uniform_on_gary(float*** filtered_img, int num_rows, int num_cols) {
//	// allocate new space
//	unsigned char*** uniform_img = new unsigned char** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		uniform_img[row] = new unsigned char* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			uniform_img[row][col] = new unsigned char[1];
//		}
//	}
//	float min_in_filtered = 100000, max_in_filtered = -100000;
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			if (filtered_img[row][col][0] < min_in_filtered) {
//				min_in_filtered = filtered_img[row][col][0];
//			}
//			if (filtered_img[row][col][0] > max_in_filtered) {
//				max_in_filtered = filtered_img[row][col][0];
//			}
//		}
//	}
//	float uniformed; 
//	unsigned char uniformed_unsigned_char;
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			uniformed = (filtered_img[row][col][0] - min_in_filtered) / (max_in_filtered - min_in_filtered) * 255;
//			uniformed_unsigned_char = uniformed;
//			uniform_img[row][col][0] = uniformed_unsigned_char;
//		}
//	}
//	return uniform_img;
//}
//
//float*** compute_gradient_magnitute(float*** horizontal_gradient, float*** vertical_gradient, int num_rows, int num_cols) {
//	// allocate new space
//	float*** gradient_magnitute = new float** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		gradient_magnitute[row] = new float* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			gradient_magnitute[row][col] = new float[1];
//		}
//	}
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			gradient_magnitute[row][col][0] = sqrt(pow(horizontal_gradient[row][col][0], 2) + pow(vertical_gradient[row][col][0], 2));
//		}
//	}
//	return gradient_magnitute;
//}
//
//unsigned char*** gray_to_binary_with_threshold(unsigned char*** img, float threshold, int num_rows, int num_cols) {
//	// allocate new space
//	unsigned char*** binary_img = new unsigned char** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		binary_img[row] = new unsigned char* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			binary_img[row][col] = new unsigned char[1];
//		}
//	}
//	float white_cnt = 0;
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			binary_img[row][col][0] = (img[row][col][0] < threshold) ? 0 : 255;
//			white_cnt += (img[row][col][0] < threshold) ? 0 : 1;
//		}
//	}
//	float percent = white_cnt / (num_rows * num_cols) * 100;
//	cout << "Threshold = " << threshold << ":white: " << percent << "%" << endl;
//	return binary_img;
//}
//
//float** index_matrix_to_threshold_matrix(int** index_matrix, int size) {
//	float** threshold_matrix = new float* [size];
//	for (int row = 0; row < size; row++) {
//		threshold_matrix[row] = new float[size];
//	}
//	for (int row = 0; row < size; row++) {
//		for (int col = 0; col < size; col++) {
//			threshold_matrix[row][col] = (index_matrix[row][col] + 0.5) / pow(size, 2) * 255;
//		}
//	}
//	return threshold_matrix;
//}
//
//unsigned char*** threshold_matrix_to_result(float** threshold_matrix, int size, unsigned char*** image, int num_rows, int num_cols) {
//	float*** uniform_image = unsigned_char_image_to_float(image, num_rows, num_cols, 1);
//	float minimum = 10000, maximum = -10000;
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			if (uniform_image[row][col][0] < minimum)
//				minimum = uniform_image[row][col][0];
//			if (uniform_image[row][col][0] > maximum)
//				maximum = uniform_image[row][col][0];
//		}
//	}
//	float dominator = 255.0 / (maximum - minimum);
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			uniform_image[row][col][0] = dominator * (uniform_image[row][col][0] - minimum);
//		}
//	}
//	unsigned char*** result = new unsigned char** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		result[row] = new unsigned char* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			result[row][col] = new unsigned char[1];
//		}
//	}
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			result[row][col][0] = (uniform_image[row][col][0] <= threshold_matrix[row % size][col % size]) ? 0 : 255;
//		}
//	}
//	return result;
//}
//
//float*** unsigned_char_image_to_float(unsigned char*** image, int num_rows, int num_cols, int num_channels) {
//	float*** float_img = new float** [num_rows];
//	for (int row = 0; row < num_rows; row++) {
//		float_img[row] = new float* [num_cols];
//		for (int col = 0; col < num_cols; col++) {
//			float_img[row][col] = new float[num_channels];
//		}
//	}
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			for (int channel = 0; channel < num_channels; channel++) {
//				float_img[row][col][channel] = image[row][col][channel];
//			}
//		}
//	}
//	return float_img;
//}
//

//

//
//unsigned char* calculate_mean_value(unsigned char* position_a, unsigned char* position_b) {
//	unsigned char* ret = new unsigned char[3];
//	for (int i = 0; i < 3; i++) {
//		ret[i] = 0.5 * (position_a[i] + position_b[i]);
//	}
//	return ret;
//}
//
//unsigned char*** mark_red_dot(unsigned char*** image, int num_rows, int num_cols, int x, int y) {
//	for (int i = y; i < y + 2; i++) {
//		for (int j = x; j < x + 2; j++) {
//			image[i][j][0] = 255;
//			image[i][j][1] = 0;
//			image[i][j][2] = 0;
//		}
//	}
//	return image;
//}
//
//vector<vector<int>> construct_STK() {
//	//Table 14.3-1 in textbook
//	vector<int> S, T, K;
//	vector<int> S1 = { 4, 1, 32, 128 };
//	S.insert(S.end(), S1.begin(), S1.end());
//
//	vector<int> S2 = { 2, 8, 16, 64 };
//	S.insert(S.end(), S2.begin(), S2.end());
//
//	vector<int> S3 = { 20, 6,3,9,40,96,192, 144 };
//	S.insert(S.end(), S3.begin(), S3.end());
//
//	vector<int> TK4 = { 18,10,72,80 };
//	T.insert(T.end(), TK4.begin(), TK4.end());
//	K.insert(K.end(), TK4.begin(), TK4.end());
//
//	vector<int> STK4 = { 148,7, 41,224 };
//	S.insert(S.end(), STK4.begin(), STK4.end());
//	T.insert(T.end(), STK4.begin(), STK4.end());
//	K.insert(K.end(), STK4.begin(), STK4.end());
//
//	vector<int> ST5_1 = { 19,146,14,84 };
//	S.insert(S.end(), ST5_1.begin(), ST5_1.end());
//	T.insert(T.end(), ST5_1.begin(), ST5_1.end());
//
//	vector<int> STK5_2 = { 11, 22,104,208 };
//	S.insert(S.end(), STK5_2.begin(), STK5_2.end());
//	T.insert(T.end(), STK5_2.begin(), STK5_2.end());
//	K.insert(K.end(), STK5_2.begin(), STK5_2.end());
//
//	vector<int> ST6 = { 147,46 };
//	S.insert(S.end(), ST6.begin(), ST6.end());
//	T.insert(T.end(), ST6.begin(), ST6.end());
//
//	vector<int> STK6 = { 23, 150,15,43,105,232,240,212 };
//	S.insert(S.end(), STK6.begin(), STK6.end());
//	T.insert(T.end(), STK6.begin(), STK6.end());
//	K.insert(K.end(), STK6.begin(), STK6.end());
//
//	vector<int> STK7 = { 151, 47,233,244 };
//	S.insert(S.end(), STK7.begin(), STK7.end());
//	T.insert(T.end(), STK7.begin(), STK7.end());
//	K.insert(K.end(), STK7.begin(), STK7.end());
//
//	vector<int> STK8 = { 107,248, 214,31 };
//	S.insert(S.end(), STK8.begin(), STK8.end());
//	T.insert(T.end(), STK8.begin(), STK8.end());
//	K.insert(K.end(), STK8.begin(), STK8.end());
//
//	vector<int> STK9 = { 215, 246,63,159,111, 235, 249,252 };
//	S.insert(S.end(), STK9.begin(), STK9.end());
//	T.insert(T.end(), STK9.begin(), STK9.end());
//	K.insert(K.end(), STK9.begin(), STK9.end());
//
//	vector<int> STK10 = { 253,247,239,191 };
//	S.insert(S.end(), STK10.begin(), STK10.end());
//	T.insert(T.end(), STK10.begin(), STK10.end());
//	K.insert(K.end(), STK10.begin(), STK10.end());
//
//	vector<int> K11 = { 254,251,223,127 };
//	K.insert(K.end(), K11.begin(), K11.end());
//	
//	//cout << "S" << endl;
//	//for (int i = 0; i < S.size(); i++) {
//	//	cout << S[i] << " ";
//	//}
//	//cout << endl;
//	//cin.get();
//	vector<vector<int>> ret;
//	ret.push_back(S);
//	ret.push_back(T);
//	ret.push_back(K);
//	return ret;
//}
//
//int check_match_ST(unsigned char*** gray_image, int r, int c) {
//	int v11 = gray_image[r - 1][c - 1][0];
//	int v12 = gray_image[r - 1][c][0];
//	int v13 = gray_image[r - 1][c + 1][0];
//	int v21 = gray_image[r][c - 1][0];
//	int v22 = gray_image[r][c][0];
//	int v23 = gray_image[r][c + 1][0];
//	int v31 = gray_image[r + 1][c - 1][0];
//	int v32 = gray_image[r + 1][c][0];
//	int v33 = gray_image[r + 1][c + 1][0];
//
//	//cout << v11 << " " << v12 << " " << v13 << endl;
//	//cout << v21 << " " << v22 << " " << v23 << endl;
//	//cout << v31 << " " << v32 << " " << v33 << endl;
//
//	//spur
//	if (v11 == 0 && v12 == 0 && v13 == 1 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//
//	//single 4-connection
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//
//	//L-cluster
//	if (v11 == 0 && v12 == 0 && v13 == 1 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 && v13 == 1 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 0 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 1 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 1 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1 && v33 == 1) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 1) {
//		return 1;
//	}
//
//	//4connected offset
//	if (v11 == 0 && v12 == 1 && v13 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 1) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 1 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//
//	// super corner cluster
//	if (v11 == 0 && v13 == 1 &&
//		v21 == 0 && v22 == 1 &&
//		v31 == 1 && v32 == 0 && v33 == 0 &&
//		v12 + v23 >= 1) {
//		return 1;
//	}
//	if (v11 == 1 && v13 == 0 &&
//		v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 1 &&
//		v12 + v21 >= 1) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 1 &&
//		v22 == 1 && v23 == 0 &&
//		v31 == 1 && v33 == 0 &&
//		v21 + v32 >= 1) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 &&
//		v31 == 0 && v33 == 1 &&
//		v23 + v32 >= 1) {
//		return 1;
//	}
//
//	//corner cluster
//	if (v11 == 1 && v12 == 1 &&
//		v21 == 1 && v22 == 1) {
//		return 1;
//	}
//
//	// tee branch
//	if (v12 == 1 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 1 &&
//		v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 1) {
//		return 1;
//	}
//	if (v12 == 0 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 1 &&
//		v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	if (v12 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v32 == 1) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v32 == 1) {
//		return 1;
//	}
//	if (v12 == 1 && 
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//
//	// vee branch
//	if (v11 == 1 && v13 == 1 &&
//		v22 == 1 &&
//		v31 + v32 + v33 >= 1) {
//		return 1;
//	}
//	if (v11 == 1 &&
//		v22 == 1 &&
//		v31 == 1 &&
//		v13 + v23 + v33 >= 1) {
//		return 1;
//	}
//	if (v22 == 1 &&
//		v31 == 1 && v33 == 1 &&
//		v11 + v12 + v13 >= 1) {
//		return 1;
//	}
//	if (v13 == 1 &&
//		v22 == 1 &&
//		v33 == 1 &&
//		v11 + v21 + v31 >= 1) {
//		return 1;
//	}
//
//	//diagonal branch
//	if (v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 1 && v32 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v32 == 0 && v33 == 1) {
//		return 1;
//	}
//	if (v12 == 0 && v13 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	return 0;
//}
//
//int check_match_K(unsigned char*** gray_image, int r, int c) {
//	int v11 = gray_image[r - 1][c - 1][0];
//	int v12 = gray_image[r - 1][c][0];
//	int v13 = gray_image[r - 1][c + 1][0];
//	int v21 = gray_image[r][c - 1][0];
//	int v22 = gray_image[r][c][0];
//	int v23 = gray_image[r][c + 1][0];
//	int v31 = gray_image[r + 1][c - 1][0];
//	int v32 = gray_image[r + 1][c][0];
//	int v33 = gray_image[r + 1][c + 1][0];
//
//	//spur
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 1) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 1 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 1 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//
//	// single 4-connection
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//
//	//L-corner
//	if (v11 == 0 && v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1 && v33 == 0) {
//		return 1;
//	}
//
//	//corner cluster
//	if (v11 == 1 && v12 == 1 && v13 == 0 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 0 && v33 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 0 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 0 && v32 == 1 && v33 == 1) {
//		return 1;
//	}
//
//	//tee branch
//	if (v12 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 1) {
//		return 1;
//	}
//	if (v12 == 1 &&
//		v21 == 1 && v22 == 1 &&
//		v32 == 1) {
//		return 1;
//	}
//	if (v21 == 1 && v22 == 1 && v23 == 1 &&
//		v32 == 1) {
//		return 1;
//	}
//	if (v12 == 1 &&
//		v22 == 1 && v23 == 1 &&
//		v32 == 1) {
//		return 1;
//	}
//
//	//vee branch
//	if (v11 == 1 && v13 == 1 &&
//		v22 == 1 &&
//		v31 + v32 + v33 >= 1) {
//		return 1;
//	}
//	if (v11 == 1 &&
//		v22 == 1 &&
//		v31 == 1 &&
//		v13 + v23 + v33 >= 1) {
//		return 1;
//	}
//	if (v11 + v12 + v13 >= 1 &&
//		v22 == 1 &&
//		v31 == 1 && v33 == 1) {
//		return 1;
//	}
//	if (v13 == 1 &&
//		v22 == 1 &&
//		v33 == 1 &&
//		v11 + v21 + v31 >= 1) {
//		return 1;
//	}
//
//	//diagonal branch
//	if (v12 == 1 && v13 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v31 == 1 && v32 == 0) {
//		return 1;
//	}
//	if (v11 == 0 && v12 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v32 == 0 && v33 == 1) {
//		return 1;
//	}
//	if (v12 == 0 && v13 == 1 &&
//		v21 == 1 && v22 == 1 && v23 == 0 &&
//		v31 == 0 && v32 == 1) {
//		return 1;
//	}
//	if (v11 == 1 && v12 == 0 &&
//		v21 == 0 && v22 == 1 && v23 == 1 &&
//		v32 == 1 && v33 == 0) {
//		return 1;
//	}
//	return 0;
//}
//
//unsigned char*** padding_image(unsigned char*** image, int num_rows, int num_cols, int num_channels) {
//	unsigned char*** ret = create_unsigned_char(num_rows + 2, num_cols + 2, num_channels);
//	for (int row = 0; row < num_rows; row++) {
//		for (int col = 0; col < num_cols; col++) {
//			for (int channel = 0; channel < num_channels; channel++) {
//				ret[row + 1][col + 1][channel] = image[row][col][channel];
//			}
//		}
//	}
//	return ret;
//}
//
//unsigned char*** STK_process(unsigned char*** image, int num_rows, int num_cols, int num_channels, int type) {
//	unsigned char*** padding = padding_image(image, num_rows, num_cols, num_channels);
//	unsigned char*** middle_map = create_unsigned_char(num_rows + 2, num_cols + 2, 1);
//	unsigned char*** ret = create_unsigned_char(num_rows, num_cols, num_channels);
//	vector<vector<int>> STK = construct_STK();
//	vector<int> S = STK[0];
//	vector<int> T = STK[1];
//	vector<int> K = STK[2];
//
//	int sum_value;
//	int if_in_STK;
//	int change_or_not;
//	int still_change = 1;
//
//	int iter = 0;
//	while (still_change != 0) {
//		iter += 1;
//		still_change = 0;
//		for (int row = 1; row < num_rows + 1; row++) {
//			for (int col = 1; col < num_cols + 1; col++) {
//				sum_value = 1 * padding[row - 1][col - 1][0] + 2 * padding[row - 1][col][0] + 4 * padding[row - 1][col + 1][0] +
//					8 * padding[row][col - 1][0] + 16 * padding[row][col + 1][0] +
//					32 * padding[row + 1][col - 1][0] + 64 * padding[row + 1][col][0] + 128 * padding[row + 1][col + 1][0];
//				if (padding[row][col][0] == 0) {
//					if_in_STK = 0;
//				}
//				else if (type == 0) {
//					if_in_STK = check_if_in_vector(S, sum_value);
//				}
//				else if (type == 1) {
//					if_in_STK = check_if_in_vector(T, sum_value);
//				}
//				else if (type == 2) {
//					if_in_STK = check_if_in_vector(K, sum_value);
//				}
//				middle_map[row][col][0] = (if_in_STK != 0) ? 1 : 0;
//			}
//		}
//		//cout << "--" << cnt1 << "--" << cnt2 << endl;
//		//unsigned char*** middle = zeroone_to_255(middle_map, num_rows + 2, num_cols + 2, 1);
//		//display_img(middle, num_rows + 2, num_cols + 2, 1, "middle");
//
//		for (int row = 1; row < num_rows + 1; row++) {
//			for (int col = 1; col < num_cols + 1; col++) {
//				if (middle_map[row][col][0] == 0) {
//					change_or_not = 0;
//				}
//				else if (type == 0 or type == 1) {
//					change_or_not = check_match_ST(middle_map, row, col);
//					if (change_or_not == 0) {
//						still_change += 1;
//						padding[row][col][0] = 0;
//					}
//				}
//				else if (type == 2) {
//					change_or_not = check_match_K(middle_map, row, col);
//					if (change_or_not == 0) {
//						still_change += 1;
//						padding[row][col][0] = 0;
//					}
//				}
//			}
//		}
//		if (iter % 30 == 0) {
//			for (int row = 1; row < num_rows + 1; row++) {
//				for (int col = 1; col < num_cols + 1; col++) {
//					ret[row - 1][col - 1][0] = padding[row][col][0];
//				}
//			}
//			unsigned char*** temp = zeroone_to_255(ret, num_rows, num_cols, num_channels);
//			display_img(temp, num_rows, num_cols, num_channels, "2a__temp");
//		}
//	}
//	for (int row = 1; row < num_rows + 1; row++) {
//		for (int col = 1; col < num_cols + 1; col++) {
//			ret[row - 1][col - 1][0] = padding[row][col][0];
//		}
//	}
//	return ret;
//}
//
//unsigned char*** gray_to_01(unsigned char*** image, int num_rows, int num_cols, int num_channels) {
//	unsigned char*** ret = create_unsigned_char(num_rows, num_cols, num_channels);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			for (int k = 0; k < num_channels; k++) {
//				ret[i][j][k] = (image[i][j][k] > 128) ? 1 : 0;
//			}
//		}
//	}
//	return ret;
//}
//
//unsigned char*** zeroone_to_255(unsigned char*** image, int num_rows, int num_cols, int num_channels) {
//	unsigned char*** ret = create_unsigned_char(num_rows, num_cols, num_channels);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			for (int k = 0; k < num_channels; k++) {
//				ret[i][j][k] = (image[i][j][k] == 1) ? 255 : 0;
//			}
//		}
//	}
//	return ret;
//}
//
//int check_if_in_vector(vector<int> vec, int num) {
//	for (int i = 0; i < vec.size(); i++) {
//		if (vec[i] == num) {
//			return 1;
//		}
//	}
//	return 0;
//}
//
//vector<int> DFS(unsigned char*** image, int num_rows, int num_cols) {
//	int * area = new int[1];
//	unsigned char*** copy = create_unsigned_char(num_rows, num_cols, 1);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			copy[i][j][0] = image[i][j][0];
//		}
//	}
//	vector<int> record;
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			if (copy[i][j][0] == 1) {
//				area[0] = 0;
//				extend_connection(copy, num_rows, num_cols, i, j, area);
//				record.push_back(area[0]);
//			}
//		}
//	}
//	return record;
//}
//
//void extend_connection(unsigned char*** image, int num_rows, int num_cols, int r, int c, int * area) {
//	image[r][c][0] = 0;
//	area[0] += 1;
//	if (r - 1 >= 0 && image[r - 1][c][0] == 1) {
//		extend_connection(image, num_rows, num_cols, r - 1, c, area);
//	}
//	if (r + 1 < num_rows && image[r + 1][c][0] == 1) {
//		extend_connection(image, num_rows, num_cols, r + 1, c, area);
//	}
//	if (c - 1 >= 0 && image[r][c - 1][0] == 1) {
//		extend_connection(image, num_rows, num_cols, r , c - 1, area);
//	}
//	if (c + 1 < num_cols && image[r][c + 1][0] == 1) {
//		extend_connection(image, num_rows, num_cols, r, c + 1, area);
//	}
//}
//
//vector<int> BFS(unsigned char*** image, int num_rows, int num_cols) {
//	int* area = new int[1];
//	unsigned char*** copy = create_unsigned_char(num_rows, num_cols, 1);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			copy[i][j][0] = image[i][j][0];
//		}
//	}
//	vector<int> record;
//	for (int r = 0; r < num_rows; r++) {
//		for (int c = 0; c < num_cols; c++) {
//			if (copy[r][c][0] == 1) {
//				area[0] = 0;
//				copy[r][c][0] = 0; // mark as visited
//				area[0] += 1;
//				queue<pair<int, int>> neighbors;
//				neighbors.push({ r, c });
//				while (!neighbors.empty()) {
//					auto rc = neighbors.front();
//					neighbors.pop();
//					int row = rc.first, col = rc.second;
//					if (row - 1 >= 0 && copy[row - 1][col][0] == 1) {
//						neighbors.push({ row - 1, col }); 
//						copy[row - 1][col][0] = 0;
//						area[0] += 1;
//					}
//					if (row + 1 < num_rows && copy[row + 1][col][0] == 1) {
//						neighbors.push({ row + 1, col }); 
//						copy[row + 1][col][0] = 0;
//						area[0] += 1;
//					}
//					if (col - 1 >= 0 && copy[row][col - 1][0] == 1) {
//						neighbors.push({ row, col - 1 }); 
//						copy[row][col - 1][0] = 0;
//						area[0] += 1;
//					}
//					if (col + 1 < num_cols && copy[row][col + 1][0] == 1) {
//						neighbors.push({ row, col + 1 }); 
//						copy[row][col + 1][0] = 0;
//						area[0] += 1;
//					}
//				}
//				record.push_back(area[0]);
//			}
//		}
//	}
//	return record;
//}
//
////Mat opencv_erosion(unsigned char*** image, int num_rows, int num_cols) {
////	Mat src = cv::imread("./results/2c__visualize_swap.jpg", cv::IMREAD_GRAYSCALE);
////	//Mat src = unsigned_img_to_mat(image, num_rows, num_cols, 1);
////	Mat dst;
////	display_img(src, "aaa");
////	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
////	erode(src, dst, element);
////	return dst;
////}
//
//unsigned char*** Mat_to_gray_unsigned_char(Mat mat) {
//	unsigned char*** ret = create_unsigned_char(mat.rows, mat.cols, 1);
//	for (int i = 0; i < mat.rows; i++) {
//		for (int j = 0; j < mat.cols; j++) {
//			ret[i][j][0] = mat.data[i * mat.cols + j];
//		}
//	}
//	return ret;
//}
//
//unsigned char*** erode(unsigned char*** image, int num_rows, int num_cols) {
//	unsigned char*** ret = copy_unsigned_char(image, num_rows, num_cols, 1);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			if (image[i][j][0] == 0) {
//				for (int x = -1; x < 2; x++) {
//					for (int y = -1; y < 2; y++) {
//						if (i + x >= 0 && i + x < num_rows && j + y >= 0 && j + y < num_cols) {
//							ret[i + x][j + y][0] = 0;
//						}
//					}
//				}
//			}
//		}
//	}
//	return ret;
//}
//
//unsigned char*** copy_unsigned_char(unsigned char*** image, int num_rows, int num_cols, int num_channels) {
//	unsigned char*** ret = create_unsigned_char(num_rows, num_cols, num_channels);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			for (int k = 0; k < num_channels; k++) {
//				ret[i][j][k] = image[i][j][k];
//			}
//		}
//	}
//	return ret;
//}
//
//unsigned char*** dilation(unsigned char*** image, int num_rows, int num_cols) {
//	unsigned char*** ret = copy_unsigned_char(image, num_rows, num_cols, 1);
//	for (int i = 0; i < num_rows; i++) {
//		for (int j = 0; j < num_cols; j++) {
//			if (image[i][j][0] == 1) {
//				for (int x = -1; x < 2; x++) {
//					for (int y = -1; y < 2; y++) {
//						if (i + x >= 0 && i + x < num_rows && j + y >= 0 && j + y < num_cols) {
//							ret[i + x][j + y][0] = 1;
//						}
//					}
//				}
//			}
//		}
//	}
//	return ret;
//}