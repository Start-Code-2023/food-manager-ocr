/**
 * @file ReceiptOCR.h
 * @brief A header for scanning and OCR (Optical Character Recognition) of receipts.
 * @author Torgrim Thorsen
 */

#include <rapidfuzz/fuzz.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <random>
#include <curl/curl.h>
#include <thread>
#include "../nlohmann/json.hpp"

using json = nlohmann::json;
using rapidfuzz::fuzz::ratio;

/**
 * @class ReceiptOCR
 * @brief A class for scanning and OCR of receipts.
 */
class ReceiptOCR {
private:
    cv::Mat frame;
    cv::Mat frameClone;
    cv::Mat receiptImage;

    enum Mode {
        DEBUG, STD
    };

    Mode mode = STD;

    struct GroundTruth {
        std::string word;
    };

    bool receiptDetected = false;
    bool scannedReceipt = false;
    bool wait = false;

    int scanPosition = 0;
    int scanDirection = 1;

    double width;
    double height;

    std::vector<cv::Point2f> sortedPoints;
    std::vector<cv::Point> largestTrapezoid;

    /**
     * @brief Undo possible "fisheye" effect as a result of perspective warp when extracting receipt from the frame image
     */
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
        if (mode == Mode::DEBUG) cv::namedWindow("Receipt Unwarped", cv::WINDOW_FREERATIO);
        if (mode == Mode::DEBUG) cv::resizeWindow("Receipt Unwarped", 1080 * 0.5, 1920 * 0.5);
        if (mode == Mode::DEBUG) cv::imshow("Receipt Unwarped", receiptImage);
    }

    /**
     * @brief Detects the receipt in the input frame.
     * @return True if a receipt is detected, false otherwise.
     */
    bool detectReceipt() {
        cv::Mat image = frame.clone();
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayImage, grayImage, cv::Size(51, 51), 0);
        cv::adaptiveThreshold(grayImage, grayImage, 128, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 59, 2);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
        cv::equalizeHist(grayImage, grayImage);
        cv::dilate(grayImage, grayImage, kernel);
        cv::erode(grayImage, grayImage, kernel);

        cv::Mat edges;
        cv::Canny(grayImage, edges, 150, 200);

        if (mode == Mode::DEBUG) cv::namedWindow("Receipt Detection (Preprocess)", cv::WINDOW_FREERATIO);
        if (mode == Mode::DEBUG) cv::resizeWindow("Receipt Detection (Preprocess)", 1080 * 0.5, 1920 * 0.5);
        if (mode == Mode::DEBUG) cv::imshow("Receipt Detection (Preprocess)", edges);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxArea = 0;

        for (const auto &contour: contours) {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);

            if (approx.size() == 4) {
                if (cv::isContourConvex(approx)) {
                    double area = cv::contourArea(approx);
                    if (area > maxArea) {
                        maxArea = area;
                        largestTrapezoid = approx;
                        sortedPoints = {};
                    }
                }
            }
        }

        if (!largestTrapezoid.empty()) {
            cv::Point2f corners[4];
            for (int i = 0; i < 4; i++) {
                corners[i] = largestTrapezoid[i];
            }

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
                                sortedPoints.push_back(topLeft);
                                sortedPoints.push_back(topRight);
                                sortedPoints.push_back(bottomLeft);
                                sortedPoints.push_back(bottomRight);
                            }
                        }
                    }
                }
            }

            frameClone = frame.clone();

            if (sortedPoints.size() == 4) {
                cv::Scalar color(255, 128, 128);
                cv::drawContours(frameClone, std::vector<std::vector<cv::Point>>{largestTrapezoid}, 0, color, 20);

                cv::Rect boundingRect = cv::boundingRect(largestTrapezoid);

                cv::Mat ReceiptROI = frame(boundingRect);

                std::vector<cv::Point2f> dstPoints;
                dstPoints.emplace_back(0, 0);
                dstPoints.emplace_back(width, 0);
                dstPoints.emplace_back(0, height);
                dstPoints.emplace_back(width, height);

                cv::Mat transformMatrix = cv::getPerspectiveTransform(sortedPoints.data(), dstPoints.data());
                cv::warpPerspective(frame, receiptImage, transformMatrix, cv::Size(int(width), int(height)));
                undoFisheye();

                if (mode == Mode::DEBUG) cv::namedWindow("Receipt", cv::WINDOW_FREERATIO);
                if (mode == Mode::DEBUG) cv::resizeWindow("Receipt", 1080 * 0.5, 1920 * 0.5);
                if (mode == Mode::DEBUG) cv::imshow("Receipt", receiptImage);

                return true;
            }
        }
        return false;
    }

    /**
     * @brief Load the JSON containing the ground truth (True String) of possible items in a receipt
     * @param jsonFileName Name and location of the ground truth JSON file
     * @return groundTruths vector of ground truth strings
     */
    static std::vector<GroundTruth> loadGroundTruths(const std::string &jsonFileName) {
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

    /**
     *
     * @param ocrOutput The OCR recognized text
     * @param truth The possible actual true value
     * @return Fuzzy Partial Token Ratio value (100 - 0)
     */
    static double checkOCR(const std::string &ocrOutput, const std::string &truth) {
        return (100 - rapidfuzz::fuzz::partial_token_ratio(ocrOutput, truth));
    }

    /**
     * @brief Performs OCR on the receipt image and sends the results via POST request to backend.
     */
    void getOCRSendPOST() {
        tesseract::TessBaseAPI tess;
        tess.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK_VERT_TEXT);
        tess.Init(nullptr, "nor");

        tess.SetVariable("tessedit_char_whitelist", "1234567890abcdefghijklmnopqrstuvwxyzæøå");

        cv::Mat receipt = receiptImage.clone();
        cv::cvtColor(receipt, receipt, cv::COLOR_BGR2GRAY); // Convert to grayscale
        cv::GaussianBlur(receipt, receipt, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(receipt, receipt, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 2);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
        cv::equalizeHist(receipt, receipt);
        cv::dilate(receipt, receipt, kernel);
        cv::erode(receipt, receipt, kernel);

        if (mode == Mode::DEBUG) cv::namedWindow("Receipt Text Extraction (Preprocess)", cv::WINDOW_FREERATIO);
        if (mode == Mode::DEBUG) cv::resizeWindow("Receipt Text Extraction (Preprocess)", 1080 * 0.5, 1920 * 0.5);
        if (mode == Mode::DEBUG) cv::imshow("Receipt Text Extraction (Preprocess)", receipt);

        std::map<std::string, std::pair<int, double>> wordInfo;

        for (int k = 0; k < 10; ++k) {
            tess.SetImage(receipt.data, receipt.cols, receipt.rows, 1,
                          int(receipt.step)); // Use 1 channel for binary image
            tess.SetSourceResolution(420);
            tess.Recognize(nullptr);

            std::string recognizedText = tess.GetUTF8Text();

            std::string jsonFileName = "../ground_truths.json";
            std::vector<GroundTruth> groundTruths = loadGroundTruths(jsonFileName);

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

        json jsonMann;

        jsonMann["user_id"] = "C8ODYfKBUkfWSpjOA8tt";

        for (const auto &entry: wordInfo) {
            const std::string &word = entry.first;
            const double confidence = entry.second.second;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> id_dist(1, 100);

            if (confidence >= 100) {
                nlohmann::json food_items = {
                        {"ID",       std::to_string(id_dist(gen))},
                        {"name",     word},
                        {"quantity", 1}
                };
                jsonMann["food_items"].push_back(food_items);
            }
        }

        jsonStr = jsonMann.dump();

        CURL *curl;
        curl_global_init(CURL_GLOBAL_ALL);
        curl = curl_easy_init();

        if (curl) {
            wait = true;

            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8080/foodmanager/v1/add/");

            struct curl_slist *headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            curl_easy_setopt(curl, CURLOPT_POST, 1);

            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.length());

            CURLcode res = curl_easy_perform(curl);

            if (res != CURLE_OK) {
                cv::Mat cleanFrame = frame.clone();
                cv::Scalar color(0, 0, 255);
                cv::drawContours(cleanFrame, std::vector<std::vector<cv::Point>>{largestTrapezoid}, 0, color, 20);
                std::string windowName = (mode == ReceiptOCR::DEBUG) ? "Receipt OCR - Debug Mode"
                                                                     : "Receipt OCR";
                cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
                cv::resizeWindow(windowName, 1080 * 0.5, 1920 * 0.5);
                cv::imshow(windowName, cleanFrame);
                std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            } else {
                cv::Mat cleanFrame = frame.clone();
                cv::Scalar color(0, 255, 0);
                cv::drawContours(cleanFrame, std::vector<std::vector<cv::Point>>{largestTrapezoid}, 0, color, 20);
                std::string windowName = (mode == ReceiptOCR::DEBUG) ? "Receipt OCR - Debug Mode"
                                                                     : "Receipt OCR";
                cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
                cv::resizeWindow(windowName, 1080 * 0.5, 1920 * 0.5);
                cv::imshow(windowName, cleanFrame);
            }

            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
    }

    /**
     * @brief Visualizes the scanning process.
     */
    void visualizeScanning() {
        if (receiptDetected && !scannedReceipt) {

            cv::Mat trailFrame = cv::Mat::zeros(cv::Size(int(width), int(height)), CV_8UC3);
            trailFrame.setTo(cv::Scalar(0, 0, 0));
            cv::Mat tempFrame = trailFrame.clone();

            for (int i = 0; i < 5; ++i) {
                cv::line(tempFrame, cv::Point(0, scanPosition - i * 50 * scanDirection),
                         cv::Point(frame.cols, scanPosition - i * 50 * scanDirection), cv::Scalar(255, 0, 0), 49);
                if (!tempFrame.empty()) cv::addWeighted(trailFrame, 1.0, tempFrame, 1 - 0.25 * i, 0.00, trailFrame);
            }

            std::vector<cv::Point2f> srcPoints;
            srcPoints.emplace_back(0, 0);
            srcPoints.emplace_back(width, 0);
            srcPoints.emplace_back(0, height);
            srcPoints.emplace_back(width, height);

            cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPoints.data(), sortedPoints.data());

            cv::Mat warpedVisualizationImage;
            cv::warpPerspective(trailFrame, warpedVisualizationImage, transformMatrix, frameClone.size());

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
    }

public:

    /**
     * @brief Parses command-line arguments and sets the operating mode.
     * @param argc Number of command-line arguments.
     * @param argv Array of command-line arguments.
     * @return 0 if successful, -1 if there is an error or help requested.
     */
    int getAppArguments(int argc, char *argv[]) {
        if (argc > 2) {
            std::cerr << "Usage: " << argv[0] << " -[command] (Optional)" << std::endl;
            return -1;
        } else if (argc == 2) {
            std::string command = argv[1];
            if (command == "-debug") {
                mode = ReceiptOCR::DEBUG;
                return 1;
            } else if (command == "-h") {
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
        return 1;
    }

    /**
     * @brief Starts the OCR and scanning process.
     * @return 0 if successful, -1 if there is an error.
     */
    int startOCR() {
        cv::VideoCapture cap(0);

        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open camera." << std::endl;
            return -1;
        }

        bool flag = true;
        bool freeze = false;
        bool scan = false;

        std::thread visualizationThread([&]() {
            while (flag) {
                if (receiptDetected && !scannedReceipt) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    getOCRSendPOST();
                    scannedReceipt = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

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
                        } else {
                            receiptDetected = false;
                        }
                    }
                } else {
                    receiptDetected = false;
                }

                visualizeScanning();

                if (!wait) {
                    std::string windowName = (mode == ReceiptOCR::DEBUG) ? "Receipt OCR - Debug Mode"
                                                                         : "Receipt OCR";
                    cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
                    cv::resizeWindow(windowName, 1080 * 0.5, 1920 * 0.5);
                    if (!frameClone.empty()) {
                        cv::imshow(windowName, frameClone);
                    }
                }

                if (key == 27) {
                    flag = false;
                }
            } else {
                std::cerr << "Error: Unable to capture frame." << std::endl;
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();
        visualizationThread.join();

        return 0;
    }
};