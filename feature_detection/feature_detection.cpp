#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
  CommandLineParser parser(argc, argv, "{@input | /home/josh/Downloads/fire.jpg | input image}" );
  Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_GRAYSCALE);
  if (src.empty())
  {
    cout << "Could not find the image \n";
    cout << "Usage: " << argv[0] << " <Input image> \n";
    return -1;
  }

  // Step 1: Detect the keypoints using FAST detector
  int threshold
  Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(50);
  
  std::vector<KeyPoint> keypoints;
  detector->detect(src, keypoints);

  // Step 2: Draw keypoints
  Mat img_keypoints;
  drawKeypoints(src, keypoints, img_keypoints);

  // Step 3: Show drawn keypoints
  imshow("FAST Keypoints", img_keypoints);

  waitKey();
  return 0;
}
#else
int main()
{
  cout << "xfeatures2d module needed, but not found \n";
  return 0;
}
#endif
