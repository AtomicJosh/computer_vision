/* This takes only the red channel of a video and saves it to a video file 
 * as well as show a few of the video's properties. This program is built 
 * to play with OpenCV's video features.
 * */

#include<opencv2/core.hpp>
#include<opencv2/videoio.hpp>
#include<iostream>

using std::cout;

int main(){
  // Load the test video
  cv::VideoCapture inputVideo("test_in.mp4");
  if(!inputVideo.isOpened()){cout<<"Couldn't load video\n"; return 0;}
  
  // Get the video codec 
  int codec = static_cast<int>(inputVideo.get(cv::CAP_PROP_FOURCC));
  cout<<"Input video codec (integer form): " << codec << "\n";
  
  // Convert video codec type to FOURCC value
  char EXT[] = {(char)(codec & 0XFF) , (char)((codec & 0XFF00) >> 8),(char)((codec & 0XFF0000) >> 16),(char)((codec & 0XFF000000) >> 24), 0};
  cout<<"Input video codec: " << EXT << "\n";
  
  // Get video dimensions
  int width = (int) inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = (int) inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
  cout<<"Input video width: " << width << "\n";
  cout<<"Input video height: " << height << "\n";

  // Get video frames per second
  int fps = (int) inputVideo.get(cv::CAP_PROP_FPS);

  // Create output video
  cv::VideoWriter outputVideo;
  outputVideo.open("test_out.mp4", codec, fps, cv::Size(width, height), true);
  if (!outputVideo.isOpened()) {cout<<"Could not open the output video\n"; return 0;}


  cv::Mat sourceVid, resultVid;
  std::vector<cv::Mat> channels; // vector of each of the channels
  for (;;) { // iterate through all inputVideo frames
    inputVideo >> sourceVid;
    if (sourceVid.empty()){break;} // check if this is the end of the video
    cv::split(sourceVid, channels); // split the frame into each of its component channels
    for(int i = 0; i < 3; ++i){ // iterate through each of the channels
      if(i!=2){ // 2 corresponds to the red channel
        channels[i] = cv::Mat::zeros(cv::Size(width, height), channels[0].type());
      }
    }
    cv::merge(channels, resultVid); // combine the channels
    outputVideo.write(resultVid); // write to file
  }
  cout << "Result with no red channel has been saved\n";
  return 0;
}
