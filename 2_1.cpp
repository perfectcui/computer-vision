
#include<opencv2/opencv.hpp>
#include<iostream>
#include<time.h>
using namespace cv;
using namespace std;

Mat process_by_at(const Mat& png1, const  Mat& png2) {
	int rows = png1.rows, cols = png1.cols;
	
	//Mat alpha(rows, cols, CV_32FC1,0);
	float a;
	Mat temp(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				//alpha.at<float>(i, j) = double(png1.at<Vec4b>(i, j)[3]) / 255;
				
				a= float(png1.at<Vec4b>(i, j)[3]) / 255.0;
				temp.at<Vec3b>(i, j)[k] =a * png1.at<Vec4b>(i, j)[k] +
					(1 - a) * png2.at<Vec3b>(i, j)[k];
			}
		}
	}
	return temp;
}
uchar* get_pixel(const Mat& png, int x, int y) {
	
	return  (uchar*)png.data + y * png.step + x * png.channels();
}
Mat process_by_point(const Mat& png1, const  Mat& png2) {
	int rows = png1.rows, cols = png1.cols;

	//Mat alpha(rows, cols, CV_32FC1,0);
	float a;
	Mat temp(rows, cols, CV_8UC3);
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < 3; k++) {
				//alpha.at<float>(i, j) = double(png1.at<Vec4b>(i, j)[3]) / 255;

				a = float(*(get_pixel(png1,i,j)+3)) / 255.0;
				*(get_pixel(temp, i, j) + k) = a * (*(get_pixel(png1, i, j) + k)) +
					(1 - a) * *(get_pixel(png2, i, j) + k);
			}
		}
	}
	return temp;
}
int main()
{

	string path1 = "E:\\计算机视觉\\exp\\1.png";
	string path2 = "E:\\计算机视觉\\exp\\R-C.png";
	clock_t s1, s2, e1, e2;
	Mat png1 = imread(path1, IMREAD_UNCHANGED), png2 = imread(path2,IMREAD_UNCHANGED);
	//imshow("2", png2);
	//while (waitKey(0) != 27);
	s1 = clock();
	Mat result1 = process_by_at(png1, png2);
	e1 = clock();
	s2 = clock();
	Mat result2 = process_by_point(png1, png2);
	e2 = clock();
	imwrite("E:\\计算机视觉\\exp\\reslut1.jpg", result1);
	imwrite("E:\\计算机视觉\\exp\\reslut2.jpg", result2);
	cout <<"time cost by at:"<< double(e1 - s1) / CLOCKS_PER_SEC << "s" << endl;
	cout <<"time cost by pointer:"<< double(e2 - s2) / CLOCKS_PER_SEC << "s" << endl;
	/*Mat t[4];
	split(png1, t);
	imshow("new_alpha", t[3]);
	while (waitKey() != 27);*/
}