#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using std::string;
using std::vector;
using std::cout;
using cv::Mat;
//using std::memcpy_s;
//using namespace cv;
//using namespace dnn;


float confThreshold = 0.9; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

vector<string> classes = { 
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
    "bear","zebra","giraffe","","backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis",
    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","","dining table","","","toilet",
    "","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};
vector<cv::Scalar> colors = {
   {0.0, 255.0, 0.0, 255.0},
   {0.0, 0.0, 255.0, 255.0},
   {255.0, 0.0, 0.0, 255.0},
   {0.0, 255.0, 255.0, 255.0},
   {255.0, 255.0, 0.0, 255.0},
   {255.0, 0.0, 255.0, 255.0},
   {80.0, 70.0, 180.0, 255.0},
   {250.0, 80.0, 190.0, 255.0},
   {245.0, 145.0, 50.0, 255.0},
   {70.0, 150.0, 250.0, 255.0},
   {50.0, 190.0, 190.0, 255.0},
}; 

cv::Mat drawBox(Mat& frame, int classId, float conf, cv::Rect box, Mat& objectMask)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = cv::max(box.y, labelSize.height);
    cv::rectangle(frame, cv::Point(box.x, box.y - round(1.5*labelSize.height)), cv::Point(box.x + round(1.5*labelSize.width), box.y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);

    cv::Scalar color = colors[classId%colors.size()];

    // Resize the mask, threshold, color and apply it on the image
    resize(objectMask, objectMask, cv::Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, color, 5, cv::LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);

    return frame;
}

Mat segmentFrame(Mat& inputFrame){
  string model = "frozen_inference_graph.pb";
  string config = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
  cv::dnn::Net net = cv::dnn::readNet(model, config);
  net.setPreferableBackend(0);
  net.setPreferableTarget(0); // 0: CPU; 1: OpenCL GPU

  Mat blob;
  cv::dnn::blobFromImage(inputFrame, // InputArray
                         blob, // OutputArray
                         (double) 1.0, // double scalefactor
                         cv::Size(inputFrame.cols, inputFrame.rows), // const Size& size
                         cv::Scalar(), // const Scalar& mean
                         true, // bool swapRB
                         false); // bool crop
  net.setInput(blob);

  vector<cv::String> outNames(2);
  outNames[0] = "detection_out_final";
  outNames[1] = "detection_masks";
  vector<Mat> outs;

  net.forward(outs, outNames);
  Mat outDetections = outs[0];
  Mat outMasks = outs[1];
  Mat mask = inputFrame;
  const int numDetections = outDetections.size[2];
  const int numClasses = outMasks.size[1];
  outDetections = outDetections.reshape(1, outDetections.total() / 7);
  for (int i = 0; i < numDetections; ++i)
  {
    float score = outDetections.at<float>(i, 2);
    if (score > confThreshold)
    {
      // Extract the bounding box
      int classId = static_cast<int>(outDetections.at<float>(i, 1));
      int left = static_cast<int>(inputFrame.cols * outDetections.at<float>(i, 3));
      int top = static_cast<int>(inputFrame.rows * outDetections.at<float>(i, 4));
      int right = static_cast<int>(inputFrame.cols * outDetections.at<float>(i, 5));
      int bottom = static_cast<int>(inputFrame.rows * outDetections.at<float>(i, 6));

      left =   cv::max(0, cv::min(left, inputFrame.cols - 1));
      top =    cv::max(0, cv::min(top, inputFrame.rows - 1));
      right =  cv::max(0, cv::min(right, inputFrame.cols - 1));
      bottom = cv::max(0, cv::min(bottom, inputFrame.rows - 1));
      cv::Rect box = cv::Rect(left, top, right - left + 1, bottom - top + 1);

      // Extract the mask for the object
      Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId)); // (int rows, int cols, int type, const Scalar& initializeElement)

      // Draw the bounding box
      mask = drawBox(inputFrame, classId, score, box, objectMask);
    }
  }
  return mask;
}

void runVideo(string filename){
  // Load the test video
  cv::VideoCapture inputVideo(filename);
  if(!inputVideo.isOpened()){cout<<"Couldn't load video\n"; return;}
  
  // Get the video codec 
  int codec = static_cast<int>(inputVideo.get(cv::CAP_PROP_FOURCC));
  
  // Convert video codec type to FOURCC value
  char EXT[] = {(char)(codec & 0XFF) , (char)((codec & 0XFF00) >> 8),(char)((codec & 0XFF0000) >> 16),(char)((codec & 0XFF000000) >> 24), 0};
  
  // Get video dimensions
  int width = (int) inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = (int) inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

  // Get video frames per second
  int fps = (int) inputVideo.get(cv::CAP_PROP_FPS);

  // Create output video
  cv::VideoWriter outputVideo;
  outputVideo.open("test_out.mp4", codec, fps, cv::Size(width, height), true);
  if (!outputVideo.isOpened()) {cout<<"Could not open the output video\n"; return;}


  Mat srcFrame, dstFrame;
  for (;;) { // iterate through all inputVideo frames
    inputVideo >> srcFrame;
    if (srcFrame.empty()){break;} // check if this is the end of the video

    dstFrame = segmentFrame(srcFrame); // run mask-rcnn on frame

    outputVideo.write(dstFrame); // write to file
  }
  cout << "Result video has been saved\n";
}

void runImage(string filename){
  Mat srcFrame, dstFrame;
  srcFrame = cv::imread(filename);
  //if (srcFrame.empty){cout<<"Could not load image\n"; return;}
  dstFrame = segmentFrame(srcFrame); // run mask-rcnn on frame
  cv::imwrite("image_out.jpg", dstFrame);
}

const char* keys = 
  "{ help h    |              | Print help message. }"
  "{ imgOrVid  | image        | Run mask-rcnn on image or video? }"
  "{ inputfile | test_img.jpg | Path to input file. }";

int main(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, keys);

  int choice = 0; // 0 for image, 1 for video
  string filename = cv::samples::findFile(parser.get<cv::String>("inputfile"));

  if (parser.get<cv::String>("imgOrVid")=="video") { runVideo(filename); }
  else if (parser.get<cv::String>("imgOrVid")=="image"){ runImage(filename); }
  else {cout<<"Invalid input\n";}
  return 0;
}
