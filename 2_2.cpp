#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
using namespace cv;
using namespace std;
Mat img;
int r = 0;
const int max_value = 100;
double fun(int x, int r) {//参数是r
	double t = 2*double(x) / 255.0 - 1;//将0-255的值，映射到-1-1
	t = 1.0 / (1.0 + exp(-r/15 * t));
	t *= 255;//将0-1的区间，映射到0-255
	return t;
}
Mat change_pixel(const Mat& img,int r) {
	Mat ans = img.clone();
	for (int i=0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			for (int k = 0; k < img.channels(); k++) {
				ans.at<Vec3b>(i, j)[k] = fun(img.at<Vec3b>(i, j)[k], r);
			}
		}
	}
	return ans;
}
void change_contrast(int ,void*) {
	Mat show = change_pixel(img, r);
	imshow("change_constract", show);
}
int main()
{
	string path = "E:\\计算机视觉\\exp\\reslut1.jpg";
	img = imread(path);
	//S形状函数，用sigmoid函数
	namedWindow("change_constract");
	createTrackbar("change_bar", "change_constract", &r, max_value, change_contrast);
	while (waitKey() != 27);
}