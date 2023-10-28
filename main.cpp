#include <cmath>
#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <fstream>
#include <curl/curl.h>
#include "nlohmann/json.hpp"
#include <rapidfuzz/fuzz.hpp>
#include "string/titlecase.h"

using json = nlohmann::json;
using rapidfuzz::fuzz::ratio;
cv::Mat frame;
cv::Mat frameClone;
cv::Mat receiptImage;

bool receiptDetected = false;
bool scannedReceipt = false;

double width;
double height;

bool debug = false;

std::vector<cv::Point2f> sortedPointsOld;

// Data structure to represent ground truths
struct GroundTruth {
    std::string word;
};

// Function to read the JSON file and store ground truths
std::vector<GroundTruth> loadGroundTruths(const std::string &jsonFileName) {
    std::vector<GroundTruth> groundTruths;

    std::ifstream file(jsonFileName);
    if (file.is_open()) {
        json j;
        file >> j;

        for (const auto &item: j) {
            GroundTruth truth;
            truth.word = item["item"];
            groundTruths.push_back(truth);
        }
    }

    return groundTruths;
}

// Function to check OCR output against ground truths and calculate confidence
double checkOCR(const std::string &ocrOutput, const std::string &truth) {
    return (100 - rapidfuzz::fuzz::partial_token_ratio(ocrOutput, truth));
}

void undoFisheye() {
    int halfWidth = receiptImage.rows / 2;
    int halfHeight = receiptImage.cols / 2;
    double strength = 1;
    double correctionRadius = sqrt(pow(receiptImage.rows, 2) + pow(receiptImage.cols, 2)) / strength;
    cv::Mat_<cv::Vec3b> Receipt = receiptImage.clone();
    cv::Mat_<cv::Vec3b> undistortedImage = receiptImage.clone();

    int newX, newY;
    double distance;
    double theta;
    double sourceX;
    double sourceY;
    double r;
    for (int i = 0; i < undistortedImage.rows; ++i) {
        for (int j = 0; j < undistortedImage.cols; j++) {
            newX = i - halfWidth;
            newY = j - halfHeight;
            distance = sqrt(pow(newX, 2) + pow(newY, 2));
            r = distance / correctionRadius;
            if (r == 0.0)
                theta = 1;
            else
                theta = atan(r) / r;

            sourceX = (halfWidth + theta * newX);
            sourceY = (halfHeight + theta * newY);

            undistortedImage(i, j)[0] = Receipt(int(sourceX), int(sourceY))[0];
            undistortedImage(i, j)[1] = Receipt(int(sourceX), int(sourceY))[1];
            undistortedImage(i, j)[2] = Receipt(int(sourceX), int(sourceY))[2];
        }
    }
    receiptImage = undistortedImage.clone();
    if (debug) cv::namedWindow("Receipt Unwarped", cv::WINDOW_FREERATIO);
    if (debug) cv::resizeWindow("Receipt Unwarped", 1080 * 0.5, 1920 * 0.5);
    if (debug) cv::imshow("Receipt Unwarped", receiptImage);
}

bool detectReceipt() {
    cv::Mat image = frame.clone();
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(grayImage, grayImage, cv::Size(31, 31), 0);
    cv::adaptiveThreshold(grayImage, grayImage, 128, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 39, 2);

    cv::Mat edges;
    cv::Canny(grayImage, edges, 50, 150);

    if (debug) cv::namedWindow("Receipt Detection (Preprocess)", cv::WINDOW_FREERATIO);
    if (debug) cv::resizeWindow("Receipt Detection (Preprocess)", 1080 * 0.5, 1920 * 0.5);
    if (debug) cv::imshow("Receipt Detection (Preprocess)", edges);

    // Find contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Iterate through the contours to find the largest rectangle (presumably the Sudoku grid)
    double maxArea = 0;
    std::vector<cv::Point> largestTrapezoid;

    // Iterate through the contours to find the largest trapezoid
    for (const auto &contour: contours) {
        // Approximate the contour with a polygon (reduce the number of vertices)
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);

        // Check if the polygon has 4 sides (quadrilateral)
        if (approx.size() == 4) {
            // Check if it's convex (to filter out concave shapes)
            if (cv::isContourConvex(approx)) {
                double area = cv::contourArea(approx);
                if (area > maxArea) {
                    maxArea = area;
                    largestTrapezoid = approx;
                    sortedPointsOld = {};
                }
            }
        }
    }

    // If a trapezoid is found, use Hough Line Transform to detect lines within it
    if (!largestTrapezoid.empty()) {
        // Extract the four corner points
        cv::Point2f corners[4];
        for (int i = 0; i < 4; i++) {
            corners[i] = largestTrapezoid[i];
        }

        // Ensure the points are in the desired order
        cv::Point2f topLeft, topRight, bottomLeft, bottomRight;
        for (const cv::Point2f &corner: corners) {
            if (corner.x < frame.cols / 2.0 && corner.y < frame.rows / 2.0) {
                topLeft = corner;
            } else if (corner.x >= frame.cols / 2.0 && corner.y < frame.rows / 2.0 && corner.x > topRight.x) {
                topRight = corner;
            } else if (corner.x < frame.cols / 2.0 && corner.y >= frame.rows / 2.0) {
                bottomLeft = corner;
            } else {
                bottomRight = corner;
            }
        }

        width = cv::norm(topLeft - topRight);
        height = cv::norm(topLeft - bottomLeft);

        if (cv::pointPolygonTest(largestTrapezoid, topLeft, false) >= 0) {
            if (cv::pointPolygonTest(largestTrapezoid, topRight, false) >= 0) {
                if (cv::pointPolygonTest(largestTrapezoid, bottomLeft, false) >= 0) {
                    if (cv::pointPolygonTest(largestTrapezoid, bottomRight, false) >= 0) {
                        if (topLeft.x != -1 && topLeft.y != -1 &&
                            topRight.x != -1 && topRight.y != -1 &&
                            bottomLeft.x != -1 && bottomLeft.y != -1 &&
                            bottomRight.x != -1 && bottomRight.y != -1) {
                            // Add the sorted corners to sortedPoints in the desired order
                            sortedPointsOld.push_back(topLeft);
                            sortedPointsOld.push_back(topRight);
                            sortedPointsOld.push_back(bottomLeft);
                            sortedPointsOld.push_back(bottomRight);
                        }
                    }
                }
            }
        }

        frameClone = frame.clone();

        // Check if there are at least 2 horizontal and 2 vertical lines
        if (sortedPointsOld.size() == 4) {
            // The trapezoid contains a grid pattern, so you can proceed to process it.
            cv::Scalar color(0, 0, 255); // Color for drawing the trapezoid
            cv::drawContours(frameClone, std::vector<std::vector<cv::Point>>{largestTrapezoid}, 0, color, 20);

            // Draw the corner points
            for (int i = 0; i < 4; i++) {
                if (debug)
                    cv::circle(frameClone, sortedPointsOld[i], 5, cv::Scalar(255, 0, 0),
                               -1); // Draw red circles at the corners
            }

            // Calculate the bounding rectangle for the largest trapezoid
            cv::Rect boundingRect = cv::boundingRect(largestTrapezoid);

            // Create a region of interest (ROI) for the bounding rectangle
            cv::Mat ReceiptROI = frame(boundingRect);

            // Define the destination points for the square
            std::vector<cv::Point2f> dstPoints;
            dstPoints.emplace_back(0, 0);
            dstPoints.emplace_back(width, 0);
            dstPoints.emplace_back(0, height);
            dstPoints.emplace_back(width, height);

            // Calculate the transformation matrix using the sorted corners
            cv::Mat transformMatrix = cv::getPerspectiveTransform(sortedPointsOld.data(), dstPoints.data());
            // Warp the largest polygon into a square
            cv::warpPerspective(frame, receiptImage, transformMatrix, cv::Size(int(width), int(height)));
            undoFisheye();
            if (debug) cv::namedWindow("Receipt", cv::WINDOW_FREERATIO);
            if (debug) cv::resizeWindow("Receipt", 1080 * 0.5, 1920 * 0.5);
            if (debug) cv::imshow("Receipt", receiptImage);
            return true;
        }
    }
    return false;
}

void generateVisualizationImage() {
    // Initialize Tesseract OCR
    tesseract::TessBaseAPI tess;
    tess.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK_VERT_TEXT);
    tess.Init(nullptr, "nor");

    tess.SetVariable("tessedit_char_whitelist", "1234567890abcdefghijklmnopqrstuvwxyzæøå");

    // Extract the receipt from the image
    cv::Mat receipt = receiptImage.clone();
    cv::cvtColor(receipt, receipt, cv::COLOR_BGR2GRAY); // Convert to grayscale
    cv::GaussianBlur(receipt, receipt, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(receipt, receipt, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 2);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
    cv::equalizeHist(receipt, receipt);
    cv::dilate(receipt, receipt, kernel);
    cv::erode(receipt, receipt, kernel);

    if (debug) cv::namedWindow("Receipt Text Extraction (Preprocess)", cv::WINDOW_FREERATIO);
    if (debug) cv::resizeWindow("Receipt Text Extraction (Preprocess)", 1080 * 0.5, 1920 * 0.5);
    if (debug) cv::imshow("Receipt Text Extraction (Preprocess)", receipt);

    std::map<std::string, std::pair<int, double>> wordInfo;

    for (int k = 0; k < 10; ++k) {
        // Perform OCR on the preprocessed cell image
        tess.SetImage(receipt.data, receipt.cols, receipt.rows, 1,
                      int(receipt.step)); // Use 1 channel for binary image
        tess.SetSourceResolution(420);
        tess.Recognize(nullptr);

        // Get the recognized text (null check)
        std::string recognizedText = tess.GetUTF8Text();

        std::string jsonFileName = "../ground_truths.json"; // Replace with your JSON file name
        std::vector<GroundTruth> groundTruths = loadGroundTruths(jsonFileName);

        // Create a map to store the confidence and occurrence count for ground truth words
        // Check the ground truth for the high-confidence word
        for (const GroundTruth &truth: groundTruths) {
            double confidence = checkOCR(recognizedText, truth.word);
            if (wordInfo.find(truth.word) == wordInfo.end()) {
                wordInfo[truth.word] = std::make_pair(1, confidence);
            } else {
                wordInfo[truth.word].first++;
                wordInfo[truth.word].second += confidence / 9;
            }
        }
    }

    std::string jsonStr;

    json jsonArray; // Create an array to hold the JSON objects

    // Iterate through wordInfo
    for (const auto &entry: wordInfo) {
        const std::string &word = entry.first;
        const double confidence = entry.second.second;

        if (confidence >= 75) {
            json jsonData;
            jsonData["Item"] = upperCase(word);
            jsonData["Score"] = confidence;

            // Add the JSON object to the array
            jsonArray.push_back(jsonData);
        }
    }

    // Serialize the JSON array to a string
    jsonStr = jsonArray.dump();

    CURL *curl;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();

    if (curl) {
        // Set the URL you want to send the JSON to
        curl_easy_setopt(curl, CURLOPT_URL, "https://example.com/api/endpoint");

        // Set headers
        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Set the request type to POST
        curl_easy_setopt(curl, CURLOPT_POST, 1);

        // Set the JSON data
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.length());

        // Perform the request
        CURLcode res = curl_easy_perform(curl);

        // Check for errors
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }

        // Clean up libcurl
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

int main(int argc, char *argv[]) {

    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " -[command] (Optional)" << std::endl;
        return -1;
    } else if (argc == 2) {
        std::string command = argv[1];
        if (command == "-debug") debug = true;
        else if (command == "-h") {
            std::cout << "Usage: " << argv[0] << " -[command] (Optional)\n" << std::endl;
            std::cout << argv[0] << ":\t Receipt OCR Scanner" << std::endl;
            std::cout << "\tEnter" << ":\t Scan Receipt\n" << std::endl;
            std::cout << "\tN/n" << ":\t New Scan" << std::endl;
            std::cout << "\tEsc" << ":\t Quit\n" << std::endl;
            std::cout << argv[0] << " -debug:\t Debug Mode - Check ComputerVision Output" << std::endl;
            return 0;
        } else {
            std::cout << "Unknown Argument: " << argv[0] << " " << argv[1] << std::endl;
            std::cerr << "Usage: " << argv[0] << " -[command] (Optional)" << std::endl;
            std::cerr << "Help: " << argv[0] << " -h" << std::endl;
            return -1;
        }
    }

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera." << std::endl;
        return -1;
    }

    bool flag = true;

    // Create a thread for generating the visualization image
    std::thread visualizationThread([&]() {
        while (flag) {
            if (receiptDetected && !scannedReceipt) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                generateVisualizationImage();
                scannedReceipt = true;
            }
            // Sleep or use some synchronization mechanism to control thread execution speed
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Adjust sleep time as needed
        }
    });

    int scanPosition = 0;
    int scanDirection = 1;

    bool freeze = false;
    bool scan = false;

    while (flag) {
        if (!freeze) {
            cap >> frame;
            cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        }

        if (!frame.empty()) {

            int key = cv::waitKey(1);

            if (key == 'n' || key == 'N') {
                freeze = false;
                scannedReceipt = false;
            }

            if (key == 13) {
                freeze = true;
                scannedReceipt = false;
                scan = true;
            }

            if (detectReceipt()) {
                if (scan) {
                    if (!receiptImage.empty()) {
                        scan = false;
                        receiptDetected = true;
                    } else receiptDetected = false;
                }
            } else {
                receiptDetected = false;
            }

            if (receiptDetected && !scannedReceipt) {

                // Create a grayscale trail frame
                cv::Mat trailFrame = cv::Mat::zeros(cv::Size(int(width), int(height)), CV_8UC3);
                trailFrame.setTo(cv::Scalar(0, 0, 0));
                cv::Mat tempFrame = trailFrame.clone();

                for (int i = 0; i < 5; ++i) {
                    cv::line(tempFrame, cv::Point(0, scanPosition - i * 50 * scanDirection),
                             cv::Point(frame.cols, scanPosition - i * 50 * scanDirection), cv::Scalar(0, 0, 255), 49);
                    if (!tempFrame.empty()) cv::addWeighted(trailFrame, 1.0, tempFrame, 1 - 0.25 * i, 0.00, trailFrame);
                }

                // Calculate the destination points for the perspective transform
                std::vector<cv::Point2f> srcPoints;
                srcPoints.emplace_back(0, 0);
                srcPoints.emplace_back(width, 0);
                srcPoints.emplace_back(0, height);
                srcPoints.emplace_back(width, height);

                // Calculate the transformation matrix using the sorted corners of the trapezoid
                cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPoints.data(), sortedPointsOld.data());

                // Warp the visualizationImage into a rectangle
                cv::Mat warpedVisualizationImage;
                cv::warpPerspective(trailFrame, warpedVisualizationImage, transformMatrix, frameClone.size());

                // Overlay the warped visualizationImage onto the frame
                if (frameClone.size() != warpedVisualizationImage.size()) {
                    std::cerr << "Error: Dimensions of 'frame' and 'warpedVisualizationImage' do not match."
                              << std::endl;
                } else if (frameClone.type() != warpedVisualizationImage.type()) {
                    std::cerr << "Error: Type of 'frame' and 'warpedVisualizationImage' do not match." << std::endl;
                    std::cerr << "Type of 'frame': " << frameClone.type() << ", Type of 'warpedVisualizationImage: "
                              << warpedVisualizationImage.type() << std::endl;
                } else {
                    cv::addWeighted(frameClone, 0.50, warpedVisualizationImage, 1.0, 0.0, frameClone);
                }

                scanPosition += 200 * scanDirection;

                if (scanPosition >= frame.rows + 150 || scanPosition <= -150) {
                    scanDirection *= -1;
                }
            }

            if (debug) {
                cv::namedWindow("Receipt OCR - Debug Mode", cv::WINDOW_FREERATIO);
                cv::resizeWindow("Receipt OCR - Debug Mode", 1080 * 0.5, 1920 * 0.5);
                cv::imshow("Receipt OCR - Debug Mode", frameClone);
            } else {
                cv::namedWindow("Receipt OCR", cv::WINDOW_FREERATIO);
                cv::resizeWindow("Receipt OCR", 1080 * 0.5, 1920 * 0.5);
                cv::imshow("Receipt OCR", frameClone);
            }

            // Check for the 'esc' key to exit the loop
            if (key == 27) {
                flag = false;
            }
        } else {
            std::cerr << "Error: Unable to capture frame." << std::endl;
            break;
        }
    }

    // Release the camera and close all OpenCV windows

    cap.release();
    cv::destroyAllWindows();
    visualizationThread.join();

    return 0;
}