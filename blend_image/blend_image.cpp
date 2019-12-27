#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

using std::cin;
using std::cout;
using std::endl;

int main(void)
{
  double alpha = 0.5; double beta; double input;

  Mat src1, src2, dst;

  cout << "Enter alpha [0.0-1.0]: " << endl;
  cin >> input;

  if (input >= 0 && input <= 1) {alpha = input;}

  src1 = imread("/home/josh/Downloads/ice.jpg");
  src2 = imread("/home/josh/Downloads/fire.jpg");

  if (src1.empty()){cout<<"Error loading src1"<<endl; return EXIT_FAILURE;}
  if (src2.empty()){cout<<"Error loading src2"<<endl; return EXIT_FAILURE;}
  
  // Make sure the images are the same dimensions, resize to largest if needed
  int newHeight = src1.rows;
  int newWidth = src1.cols;
  bool resize1 = false, resize2 = false;
  if (src1.rows<src2.rows){newHeight=src2.rows; resize1=true;}
  if (src1.rows>src2.rows){newHeight=src1.rows; resize2=true;}
  if (src1.cols<src2.cols){newWidth=src2.cols; resize1=true;}
  if (src1.cols>src2.cols){newWidth=src1.cols; resize2=true;}

  if (resize1) {cv::resize(src1, src1, Size(newWidth, newHeight), 0, 0, INTER_CUBIC);}
  if (resize2) {cv::resize(src2, src2, Size(newWidth, newHeight), 0, 0, INTER_CUBIC);}

  // merge the two images
  beta = (1.0 - alpha);
  addWeighted(src1, alpha, src2, beta, 0.0, dst);

  // save the image
  imwrite("/home/josh/Downloads/fire_and_ice.jpg", dst);

  // show the merged image
  imshow("Linear Blend", dst);
  waitKey(0);

  return 0;
}
