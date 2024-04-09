#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include<time.h>
using namespace std;
using namespace cv;
Mat change(Mat &img) {
	Mat ans = img.clone();
	double w = img.cols, h = img.rows;
	w *= 0.5, h *= 0.5;
	//这是对于ans的遍历
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			//将坐标转化为中心坐标
			double x_c = x / w - 1, y_c = y / h - 1;
			double r = x_c * x_c + y_c * y_c;
			if (r <= 1) {
				r = pow(r, 0.5);
				double theta = pow(1 - r, 2);
				double x_n = cos(theta) * x_c - sin(theta) * y_c;
				double y_n = cos(theta) * y_c + sin(theta) * x_c;
				//将中心坐标转化为像素坐标
				x_n = (x_n + 1) * w;
				y_n = (y_n + 1) * h;
				ans.at<Vec3b>(y, x) = img.at<Vec3b>(y_n, x_n);
			}
			//否则r>1时，新图（x,y）与原本（x,y）对应，不用处理
		}
	}
	return ans;
}

int main() {
	VideoCapture  s(0);
	if (!s.isOpened()) {
		cout << "error:could not open camera" << endl;
		return -1;
	}
	namedWindow("camera", WINDOW_AUTOSIZE);
	int delay = 1;
	clock_t start, end;
	Size  size = Size(int(s.get(CAP_PROP_FRAME_WIDTH)), int(s.get(CAP_PROP_FRAME_HEIGHT)));
	VideoWriter writer(".\\out.mp4",VideoWriter::fourcc('M','P','4','V'),30,size);
	while (waitKey(delay) != 27) {
		//从摄像头读取一帧
		start = clock();
		Mat img;
		s >> img;
		if (img.empty()) {
			cout << "Can not get frame.exit!" << endl;
		}
		start = clock();
		Mat result = change(img);
		//end = clock();
		//cout << "change image cost " << end - start <<"/" << CLOCKS_PER_SEC << "s" << endl;
		//cout << "change image framge/s " << CLOCKS_PER_SEC /(end-start)<< endl;
		writer.write(result);
		imshow("camera", result);
		end = clock();
		cout << "change image cost " << end - start << "/" << CLOCKS_PER_SEC << "s" << endl;
		cout << "change image framge/s " << CLOCKS_PER_SEC / (end - start) << endl;
	}
	s.release();
	writer.release();
	destroyAllWindows();
}