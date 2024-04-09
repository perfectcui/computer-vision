#include<opencv2/opencv.hpp>
#include<iostream>
#include<time.h>
using namespace cv;
using namespace std;
//将两个灰色图像按像素相乘
Mat mutiple(const Mat& m1, const Mat& m2) {
	Mat ans = m1.clone();
	int w = m1.cols, h = m1.rows;
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			ans.at<float>(y, x) *= m2.at<float>(y, x);
		}
	}
	return ans;
}

void circle_on_picture(Mat& m, const Point& p) {
	//颜色默认,半径为6，宽度为1
	cv::Scalar color(0, 10, 255);
	int r = 6;
	int w = 1;
	circle(m, p, r, color, w);
}
void show_circle(Mat& put, Mat& ans, float t) {
	Mat after;
	//在图像上画圈
	normalize(ans, after, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	for (int x = 0; x < put.cols; x++) {
		for (int y = 0; y < put.rows; y++) {
			if (after.at<float>(y, x) > t) {
				circle_on_picture(put, Point(x, y));
			}
		}
	}
}
//将float图像转化为unchar图像，并展示
void show_float_picture(Mat& put,string name) {
	Mat ans = put.clone();
	convertScaleAbs(ans, ans);
	imshow(name, ans);
}
//根据阈值，绘制标注后的图片
Mat show_picture(const Mat& put,Mat&ans, int r ,string name="ssd") {
	//阈值为100
	Mat temp = put.clone();
	show_circle(temp,ans, r);
	convertScaleAbs(temp, temp);
	/*imshow(name,temp);*/
	return temp;
}
void my_cornerharris(const Mat&input,Mat&dst,int block_size,int k_size,float alpha) {
	Mat dx, dy;
	Sobel(input, dx, CV_32F, 1, 0, k_size);
	Sobel(input, dy, CV_32F, 0, 1, k_size);
	Mat dxx = mutiple(dx, dx);
	Mat dyy = mutiple(dy, dy);
	Mat dxy = mutiple(dx, dy);
	dst = input.clone();
	double r = (k_size + 1) / 6;//高斯函数的标准差
	GaussianBlur(dxx, dxx, Size(block_size, block_size),r,r);
	GaussianBlur(dyy, dyy, Size(block_size, block_size), r, r);
	GaussianBlur(dxy, dxy, Size(block_size, block_size), r, r);
	for (int x = 0; x < input.cols; x++) {
		for (int y = 0; y < input.rows; y++) {
			dst.at<float>(y, x) = dxx.at<float>(y, x) * dyy.at<float>(y, x) - pow(dxy.at<float>(y, x), 2) - alpha * pow(dxx.at<float>(y, x) + dyy.at<float>(y, x), 2);
		}
	}
	/*convertScaleAbs(dxx, dxx);
	imshow("dxx", dxx);
	convertScaleAbs(dyy, dyy);
	imshow("dyy", dyy);
	convertScaleAbs(dxy, dxy);
	imshow("dxy", dxy);*/
}
int r1 = 0,r2=0;
const int max_value = 255;
Mat m1, m2,a1,a2;
void change_contrast(int, void*) {
	Mat show = show_picture(m1, a1, r1);
	imshow("opencv_ans", show);
}
void change_contrast2(int, void*) {
	Mat show = show_picture(m2, a2, r2);
	imshow("my_ans", show);
}
int main() {
	string path = "E:\\计算机视觉\\exp\\house.jpg";
	Mat input = imread(path);
	resize(input, input, Size(1200, 700));
	Mat grey;
	cvtColor(input, grey,COLOR_BGR2GRAY);
	imshow("gray_put", grey);
	Mat grey_float32;
	grey.convertTo(grey_float32,CV_32F);//将图像转化为灰色图
	Mat grey_float32_copy = grey_float32.clone();
	Mat opencv_ans,my_ans;
	clock_t s1, s2, e1, e2;
	//边界都选用的boderdefault
	s1 = clock();
	cornerHarris(grey_float32, opencv_ans, 7, 3, 0.04);
	/*show_float_picture(opencv_ans, "opencv_temp");*/
	e1 = clock();
	/*show_picture(grey_float32, opencv_ans, "cv_ans");*/
	s2 = clock();
	my_cornerharris(grey_float32_copy, my_ans, 7, 3, 0.04);
	//show_float_picture(my_ans, "my_temp");
	e2 = clock();
	/*show_picture(grey_float32_copy, my_ans, "my_ans");*/
	cout <<"opencv harris cost " << e1 - s1 << "us" << endl;
	cout << "my harris cost " << e2 - s2 << "us" << endl;
	m1 = grey_float32, m2 = grey_float32_copy;
	a1 = opencv_ans, a2 = my_ans;
	namedWindow("opencv_ans");
	createTrackbar("threshold", "opencv_ans", &r1, max_value, change_contrast);
	namedWindow("my_ans");
	createTrackbar("threshold", "my_ans", &r2, max_value, change_contrast2);
	while (waitKey() != 27);
}