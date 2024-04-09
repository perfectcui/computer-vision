#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<vector>
#include<time.h>
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
	sigma *= 2 * sigma;//2*sigma^2
	int mid = length / 2;
	for (int i = 0; i < length; i++) {
		double dis = mid - i;
		kernel[i] = exp(-dis * dis / sigma);
	}
	//计算归一化因子
	double sums = 0;
	//sums = accumulate(kernel.begin(), kernel.end(), 0);
	sums = accumulate(kernel.begin(), kernel.end(), 0.0);
	//for (auto i : kernel)
	//	sums += i;
	Mat temp = output.clone();
	//对图像的边界进行处理，牺牲速度，不进行填充
	//纵向进行核操作
	for (int x = 0; x < input.cols; x++) {
		for (int y = 0; y < input.rows; y++) {
			Vec3b t(0, 0, 0);
			for (int j = 0, index = y - mid; j < length; j++, index++) {
				if (index >= 0 && index < input.rows) {
					t += kernel[j] / sums * input.at<Vec3b>(index, x);
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
			for (int j = 0, index = x - mid; j < length; j++, index++) {
				if (index >= 0 && index < input.cols) {
					t += kernel[j] / sums * temp.at<Vec3b>(y, index);
				}
			}
			output.at<Vec3b>(y, x) = t;
		}
	}
}
//进行加速，可以保存一个颜色模版，因为颜色值的差距最大255-0
//依据r生成颜色值的查看表
void make_color_look_up(double r,vector<double>&look_up) {
	r = 2 * r * r;//r=2*r^2
	for (int i = 0; i <= 255; i++) {
		double value = exp(-i * i / r);
		look_up[i] = value;
	}
}
//生成一个基于距离的二维高斯滤波核
vector<vector<double>>make_guss_kernal(double r,int &len) {
	int length = 6 * r - 1;
	if (length < 1)
		length = 1;
	len = length;
	r = 2 * r * r;
	vector<vector<double>>kernal(length, vector<double>(length));
	int mid = length / 2;
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < length; j++) {
			kernal[i][j] = exp((-(i - mid) * (i - mid) - (j - mid) * (j - mid)) / r);
		}
	}
	return kernal;
}
//需要传入颜色和空间滤波的sigma
Mat Bilateral_Filter(const Mat& input, double sigmap, double sigmac) {
	vector<double>colors(256);
	make_color_look_up(sigmac, colors);
	int len;
	vector<vector<double>>kernal = make_guss_kernal(sigmap, len);
	//对图片进行填充,镜像复制
	Mat true_put;
cv:copyMakeBorder(input, true_put, len / 2, len / 2, len / 2, len / 2, cv::BORDER_REFLECT);
	Mat ans = input.clone();
	for (int x = 0; x < input.cols; x++) {
		for (int y = 0; y < input.rows; y++) {
			int c_x = x + len / 2;
			int c_y = y + len / 2;
			for (int c = 0; c < input.channels(); c++) {
				//获取该通道的中心颜色值
				int color = true_put.at<Vec3b>(c_y, c_x)[c];
				double c_t = 0;
				double sums = 0;
				for (int n = x; n < x + len; n++) {
					for (int m = y; m < y + len; m++) {
						int c_now = true_put.at<Vec3b>(m, n)[c];
						double t = colors[abs(c_now - color)] * kernal[m-y][n - x];
						c_t += t * c_now;
						sums += t;
					}
				}
				ans.at<Vec3b>(y, x)[c] = c_t / sums;
			}
		}
	}
	return ans;
}

Mat add_noise(Mat& image) {
	int rows = image.rows;
	int cols = image.cols;

	// 创建一个与原始图像大小和类型相同的噪声图像  
	cv::Mat noise(rows, cols, image.type());

	// 设置随机数种子，以便每次运行程序时生成的噪声都不同  
	std::srand(std::time(0));
	double mean = 0.0;
	double stddev = 20; // 标准差
cv:randn(noise, mean, stddev);

	// 将噪声添加到原始图像中  
	cv::Mat noisyImage = image + noise;

	// 确保像素值在有效范围内（0-255）  
	noisyImage.convertTo(noisyImage, CV_8U);
	return noisyImage;
}
int main() {
	//string file_path = "E:\\计算机视觉\\exp\\reslut1.jpg";
	string file_path = "E:\\计算机视觉\\exp\\gass.jpg";
	Mat input = imread(file_path);
	Mat noisy = add_noise(input);
	Mat output = input.clone();
	double sigma = 2;
	clock_t s1, s2, e1, e2;
	s1=clock();
	gaussianBlur(noisy, output, sigma);
	e1 = clock();
	cout << "my guass fileter cost :" << (e1 - s1) << "clock" << endl;
	s1 = clock();
	Mat bians = Bilateral_Filter(noisy, sigma, sigma/4);
	e1 = clock();
	s2 = clock();
	Mat cvans;
	cv::bilateralFilter(noisy, cvans, 0, sigma, sigma/4, cv::BORDER_REFLECT);
	e2 = clock();
	cout << "my bilateral fileter cost :" << (e1 - s1) << "clock"<<endl;
	cout << "cv bilateral fileter cost :" << (e2 - s2) << "clock"<<endl;
	imshow("primary", input);
	imshow("noisy", noisy);
	imshow("my_guass", output);
	imshow("my_bi", bians);
	imshow("cv:ans", cvans);

	while (waitKey() != 27);
}