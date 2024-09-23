#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int main() {
    // Load the YOLOv8 model
    std::string modelPath = "yolov8x.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // Check if CUDA is available and set the preferable backend and target accordingly
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        std::cerr << "Error: No CUDA-enabled device found." << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat image = cv::imread("bus.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Prepare the image for object detection
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
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
                int centerX = (int)(data[0] * image.cols);
                int centerY = (int)(data[1] * image.rows);
                int width = (int)(data[2] * image.cols);
                int height = (int)(data[3] * image.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                // Draw bounding box
                cv::rectangle(image, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 3);
            }
        }
    }

    // Save the image to a PNG file
    cv::imwrite("output.png", image);

    return 0;
}
