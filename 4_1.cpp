#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<vector>
//#include<algorithm>
#include<numeric>
using namespace std;
using namespace cv;

void gaussianBlur(const Mat& input, Mat& output, double sigma) {
	int length = 6 * sigma - 1;//单维度卷积核的长度
	//对卷积核内的元素值设置为浮点数
	//默认最小的核长度为1
	if (length < 1)
		length = 1;
	vector<double>kernel(length);
	sigma *= 2*sigma;//2*sigma^2
	int mid = length / 2;
	for (int i = 0; i<length; i++) {
		double dis = mid - i;
		kernel[i] = exp(-dis * dis / sigma);
	}
	//计算归一化因子
	double sums =0;
	//sums = accumulate(kernel.begin(), kernel.end(), 0);
	sums = accumulate(kernel.begin(), kernel.end(), 0.0);
	//for (auto i : kernel)
	//	sums += i;
	Mat temp = output.clone();
	//对图像的边界进行处理，牺牲速度，不进行填充
	//纵向进行核操作
	for (int x = 0; x < input.cols; x++) {
		for (int y = 0; y < input.rows; y++) {
			Vec3b t(0,0,0);
			for (int j =0,index= y - mid; j < length; j++,index++) {
				if (index>= 0 && index < input.rows) {
					t += kernel[j]/sums * input.at<Vec3b>(index, x);
				}
			}
			temp.at<Vec3b>(y, x) = t;
		}
	}
	//imshow("temp", temp);
	//while (waitKey() != 27);
	////横向
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			Vec3b t(0, 0, 0);
			for (int j =0,index= x - mid; j < length; j++,index++) {
				if (index >= 0 && index < input.cols) {
					t += kernel[j] / sums * temp.at<Vec3b>(y, index);
				}
			}
			output.at<Vec3b>(y, x) = t;
		}
	}
}
Mat add_noise(Mat& image) {
	int rows = image.rows;
	int cols = image.cols;

	// 创建一个与原始图像大小和类型相同的噪声图像  
	cv::Mat noise(rows, cols, image.type());

	// 设置随机数种子，以便每次运行程序时生成的噪声都不同  
	std::srand(std::time(0));
	double mean = 0.0;
	double stddev = 40; // 标准差
	cv:randn(noise, mean, stddev);

	// 将噪声添加到原始图像中  
	cv::Mat noisyImage = image + noise;

	// 确保像素值在有效范围内（0-255）  
	noisyImage.convertTo(noisyImage, CV_8U);
	return noisyImage;
}
int main() {
	string file_path = "E:\\计算机视觉\\exp\\reslut1.jpg";
	//string file_path = "E:\\计算机视觉\\exp\\gass.jpg";
	Mat input = imread(file_path);
	Mat noisy = add_noise(input);
	Mat output = input.clone();
	double sigma = 1.7;
	gaussianBlur(noisy, output, sigma);
	imshow("primary", input);
	imshow("noisy", noisy);
	imshow("result", output);
	while (waitKey() != 27);
}