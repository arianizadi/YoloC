#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int main() {
  // Open the default camera
  cv::VideoCapture cap("http://192.168.1.21:8000/");
  if(!cap.isOpened()) {
    std::cerr << "Error: Could not open the camera." << std::endl;
    return -1;
  }

  // Load pre-trained YOLOv8 model from ONNX file for object detection
  std::string modelPath = "/home/outkast/YoloC/yolov8x.onnx";
  cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

  cv::Mat frame;
  while(true) {
    // Capture frame-by-frame
    cap >> frame;

    // If the frame is empty, break immediately
    if(frame.empty()) {
      std::cerr << "Error: Blank frame grabbed." << std::endl;
      break;
    }

    // Prepare the frame for object detection
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Run forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // Post-process the output to get bounding boxes
    float confThreshold = 0.5;
    for (size_t i = 0; i < outs.size(); ++i) {
      float* data = (float*)outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point classIdPoint;
        double confidence;
        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        if (confidence > confThreshold) {
          int centerX = (int)(data[0] * frame.cols);
          int centerY = (int)(data[1] * frame.rows);
          int width = (int)(data[2] * frame.cols);
          int height = (int)(data[3] * frame.rows);
          int left = centerX - width / 2;
          int top = centerY - height / 2;

          // Draw bounding box
          cv::rectangle(frame, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 3);
        }
      }
    }

    // Display the resulting frame
    cv::imshow("Webcam", frame);

    // Press 'q' to exit the loop
    if(cv::waitKey(30) >= 0) {
      break;
    }
  }

  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  cv::destroyAllWindows();

  return 0;
}
