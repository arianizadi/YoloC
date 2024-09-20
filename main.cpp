#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  // Open the default camera
  cv::VideoCapture cap("http://192.168.1.21:8000/");
  if(!cap.isOpened()) {
    std::cerr << "Error: Could not open the camera." << std::endl;
    return -1;
  }

  cv::Mat frame;
  while(true) {
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if(frame.empty()) {
      std::cerr << "Error: Blank frame grabbed." << std::endl;
      break;
    }

    // Display the resulting frame
    cv::imshow("Webcam", frame);

    // Press 'q' to exit the loop
    if(cv::waitKey(30) >= 0)
      break;
  }

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  cv::destroyAllWindows();

  return 0;
}
