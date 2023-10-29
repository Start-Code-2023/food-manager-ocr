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
// Declare Private class members, not visible outside the header file
private:
    cv::Mat frame;          ///< Input frame from the camera
    cv::Mat frameClone;     ///< Copy of the input frame for processing
    cv::Mat receiptImage;   ///< Extracted receipt image

    enum Mode {
        DEBUG, STD
    };

    Mode mode = STD;         ///< Operating mode (STD or DEBUG)

    struct GroundTruth {
        std::string word;
    };

    bool receiptDetected = false;   ///< Flag to indicate if a receipt is detected
    bool scannedReceipt = false;    ///< Flag to indicate if receipt is scanned and OCR is performed
    bool wait = false;              ///< Flag for waiting during the OCR POST process

    int scanPosition = 0;           ///< Current position of the visualized scanner
    int scanDirection = 1;          ///< Direction of scanning (up or down)

    double width;                   ///< Width of the detected receipt
    double height;                  ///< Height of the detected receipt

    std::vector<cv::Point2f> sortedPoints;      ///< Sorted corners of the detected receipt
    std::vector<cv::Point> largestTrapezoid;    ///< Points of the largest trapezoidal contour (Most Likely the receipt)

    /**
     * @brief Undo possible "fisheye" effect as a result of perspective warp when extracting receipt from the frame image
     */
    void undoFisheye() {
        // Get center of the receipt image
        int halfWidth = receiptImage.rows / 2;
        int halfHeight = receiptImage.cols / 2;

        // Fisheye reversal strength
        double strength = 1;

        // Calculate the Fisheye corrective radius
        double correctionRadius = sqrt(pow(receiptImage.rows, 2) + pow(receiptImage.cols, 2)) / strength;

        // Clone the receipt Image for processing
        cv::Mat_<cv::Vec3b> Receipt = receiptImage.clone();
        cv::Mat_<cv::Vec3b> undistortedImage = receiptImage.clone();

        // Undo Fisheye effect
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
        // Update the receipt Image with the unwarped image
        receiptImage = undistortedImage.clone();

        // If in DEBUG Mode show the unwarped image
        if (mode == Mode::DEBUG) cv::namedWindow("Receipt Unwarped", cv::WINDOW_FREERATIO);
        if (mode == Mode::DEBUG) cv::resizeWindow("Receipt Unwarped", 1080 * 0.5, 1920 * 0.5);
        if (mode == Mode::DEBUG) cv::imshow("Receipt Unwarped", receiptImage);
    }

    /**
     * @brief Detects the receipt in the input frame.
     * @return True if a receipt is detected, false otherwise.
     */
    bool detectReceipt() {
        // Clone the input frame
        cv::Mat image = frame.clone();

        // Preprocess the Image for contour detection
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayImage, grayImage, cv::Size(51, 51), 0);
        cv::adaptiveThreshold(grayImage, grayImage, 128, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 59, 2);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
        cv::equalizeHist(grayImage, grayImage);
        cv::dilate(grayImage, grayImage, kernel);
        cv::erode(grayImage, grayImage, kernel);

        cv::Mat edges;
        cv::Canny(grayImage, edges, 50, 150);

        // if in DEBUG Mode show the preprocessed image
        if (mode == Mode::DEBUG) cv::namedWindow("Receipt Detection (Preprocess)", cv::WINDOW_FREERATIO);
        if (mode == Mode::DEBUG) cv::resizeWindow("Receipt Detection (Preprocess)", 1080 * 0.5, 1920 * 0.5);
        if (mode == Mode::DEBUG) cv::imshow("Receipt Detection (Preprocess)", edges);

        // Initialize a vector to hold the detected contours
        std::vector<std::vector<cv::Point>> contours;
        // Find contours in the preprocessed image
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Initialize a starting Area Value of 0
        double maxArea = 0;

        // Iterate over the contours
        for (const auto &contour: contours) {
            // initialize a vector of points
            std::vector<cv::Point> approx;
            // Check if the contours approximately form polygons
            cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);

            // check if the polygon is rectangular/Trapezoidal
            if (approx.size() == 4) {
                // confirm that the contour in convex to filter out concave shapes
                if (cv::isContourConvex(approx)) {
                    // Get the area of the contour
                    double area = cv::contourArea(approx);
                    // If the area of the contour is larger than the previous detected polygon/trapezoid
                    // update the maxArea Value, LargestTrapezoid and the sortedPoints vector
                    if (area > maxArea) {
                        maxArea = area;
                        largestTrapezoid = approx;
                        sortedPoints = {};
                    }
                }
            }
        }

        // Check that the trapezoid is not empty
        if (!largestTrapezoid.empty()) {
            // Initialize a c-style array to hold the corners of the trapezoid
            cv::Point2f corners[4];
            // Iterate over the corners of the trapezoid and push them to the corner vector
            for (int i = 0; i < 4; i++) {
                corners[i] = largestTrapezoid[i];
            }

            // Initialize specific corner points
            cv::Point2f topLeft, topRight, bottomLeft, bottomRight;
            // Iterate over the corners and make sure they are in the correct order
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

            // Store the height and width of the detected trapezoid
            width = cv::norm(topLeft - topRight);
            height = cv::norm(topLeft - bottomLeft);

            // Double-check that the corners are within the detected receipt ROI (Region of Interest)
            if (cv::pointPolygonTest(largestTrapezoid, topLeft, false) >= 0) {
                if (cv::pointPolygonTest(largestTrapezoid, topRight, false) >= 0) {
                    if (cv::pointPolygonTest(largestTrapezoid, bottomLeft, false) >= 0) {
                        if (cv::pointPolygonTest(largestTrapezoid, bottomRight, false) >= 0) {
                            if (topLeft.x != -1 && topLeft.y != -1 &&
                                topRight.x != -1 && topRight.y != -1 &&
                                bottomLeft.x != -1 && bottomLeft.y != -1 &&
                                bottomRight.x != -1 && bottomRight.y != -1) {
                                // if the corners pass the test push them to the sortedPoints Vector in order
                                sortedPoints.push_back(topLeft);
                                sortedPoints.push_back(topRight);
                                sortedPoints.push_back(bottomLeft);
                                sortedPoints.push_back(bottomRight);
                            }
                        }
                    }
                }
            }

            // Clone the input frame again for further processing
            frameClone = frame.clone();

            // Check that there are four corners in the sortedPoints vector
            if (sortedPoints.size() == 4) {
                // Draw the contour of the detected trapezoid in Yellow
                cv::Scalar color(0, 255, 255);
                cv::drawContours(frameClone, std::vector<std::vector<cv::Point>>{largestTrapezoid}, 0, color, 10);

                // Initialize and construct destination points based on the width and height of the detected trapezoid
                // for the perspective warping of the receipt to a squared image
                std::vector<cv::Point2f> dstPoints;
                dstPoints.emplace_back(0, 0);
                dstPoints.emplace_back(width, 0);
                dstPoints.emplace_back(0, height);
                dstPoints.emplace_back(width, height);

                // Get the perspective transformation matrix to perform perspective warp based on the destination and the destination points
                cv::Mat transformMatrix = cv::getPerspectiveTransform(sortedPoints.data(), dstPoints.data());
                // Warp and crop the input frame down to a squared image of the detected receipt
                cv::warpPerspective(frame, receiptImage, transformMatrix, cv::Size(int(width), int(height)));

                // If in DEBUG Mode show the extracted image
                if (mode == Mode::DEBUG) cv::namedWindow("Receipt", cv::WINDOW_FREERATIO);
                if (mode == Mode::DEBUG) cv::resizeWindow("Receipt", 1080 * 0.5, 1920 * 0.5);
                if (mode == Mode::DEBUG) cv::imshow("Receipt", receiptImage);

                // Undo possible fisheye effect that might have occured due to the perspective warp to a squared image
                undoFisheye();

                // Return that a receipt was, most likely, found
                return true;
            }
        }
        // If the trapezoid was empty no receipt was detected
        return false;
    }

    /**
     * @brief Load the JSON containing the ground truth (True String) of possible items in a receipt
     * @param jsonFileName Name and location of the ground truth JSON file
     * @return groundTruths vector of ground truth strings
     */
    static std::vector<GroundTruth> loadGroundTruths(const std::string &jsonFileName) {
        // Initialize a Vector of Ground truths (Correct spelling of items)
        std::vector<GroundTruth> groundTruths;

        // Load the JSON with the ground truths and push them to the vector
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

        // Return the vector of ground truths
        return groundTruths;
    }

    /**
     *
     * @param ocrOutput The OCR recognized text
     * @param truth The possible actual true value
     * @return Fuzzy Partial Token Ratio value (100 - 0)
     */
    static double checkOCR(const std::string &ocrOutput, const std::string &truth) {
        // Check if the OCR output is a fuzzy (Partial) match to the ground truth and return a score
        return rapidfuzz::fuzz::partial_token_set_ratio(ocrOutput, truth);
    }

    /**
     * @brief Performs OCR on the receipt image and sends the results via POST request to backend.
     */
    void getOCRSendPOST() {
        // Initialize the OCR Model
        tesseract::TessBaseAPI tess;
        tess.SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK_VERT_TEXT);
        tess.Init(nullptr, "nor");

        // Whitelist specific characters for the OCR Model
        tess.SetVariable("tessedit_char_whitelist", "1234567890abcdefghijklmnopqrstuvwxyzæøå");

        // Clone the receipt image for processing (Make the image cleaner for OCR detection)
        cv::Mat receipt = receiptImage.clone();
        cv::cvtColor(receipt, receipt, cv::COLOR_BGR2GRAY); // Convert to grayscale
        cv::GaussianBlur(receipt, receipt, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(receipt, receipt, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 2);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
        cv::equalizeHist(receipt, receipt);
        cv::dilate(receipt, receipt, kernel);
        cv::erode(receipt, receipt, kernel);

        // If in DEBUG Mode show the processed receipt image
        if (mode == Mode::DEBUG) cv::namedWindow("Receipt Text Extraction (Preprocess)", cv::WINDOW_FREERATIO);
        if (mode == Mode::DEBUG) cv::resizeWindow("Receipt Text Extraction (Preprocess)", 1080 * 0.5, 1920 * 0.5);
        if (mode == Mode::DEBUG) cv::imshow("Receipt Text Extraction (Preprocess)", receipt);

        // Initialize a map containing a string (the Word) and a pair (Occurrences and their score)
        std::map<std::string, std::pair<int, double>> wordInfo;

        // Set up a for loop to do ten OCR passes
        for (int k = 0; k < 10; ++k) {
            // Give the OCR Model the preprocessed receipt image
            tess.SetImage(receipt.data, receipt.cols, receipt.rows, 1,
                          int(receipt.step));
            // Specify the receipt Image resolution
            tess.SetSourceResolution(420);
            // Run the OCR model on the Image
            tess.Recognize(nullptr);

            // Extract the recognized text from the OCR model as UTF-8 encoded string
            std::string recognizedText = tess.GetUTF8Text();

            // Specify the location of the ground truth JSON  and load them into a vector
            std::string jsonFileName = "../ground_truths.json";
            std::vector<GroundTruth> groundTruths = loadGroundTruths(jsonFileName);

            // Iterate over each word in the ground truth vector and get the fuzzy score
            // Update the map with the ground truth word, occurrences and score
            for (const GroundTruth &truth: groundTruths) {
                double confidence = checkOCR(recognizedText, truth.word);
                // Check if we have encountered this word before, if not add it to the map
                // if we have encountered it update it,
                if (wordInfo.find(truth.word) == wordInfo.end()) {
                    wordInfo[truth.word] = std::make_pair(1, confidence);
                } else {
                    wordInfo[truth.word].first++;
                    wordInfo[truth.word].second += confidence / 10;
                }
            }
        }

        // Initialize a JSON string
        std::string jsonStr;

        // Initialize nlohmann/json for creating JSON
        json jsonMann;

        // Add an item to the JSON with the user id for sending the JSON to the backend API
        jsonMann["user_id"] = "6fkM3h0Y8DMvemRlX5eu";

        // Iterate over the words we encountered in the OCR
        for (const auto &entry: wordInfo) {
            // Extract the word and score
            const std::string &word = entry.first;
            const double confidence = entry.second.second;

            // Initialize Random ID Generator (For DEMO Purpose)
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> id_dist(1, 100);

            // Check if the score is equal to or above  90
            // if it is we add the item to the JSON we send to the backend
            if (confidence >= 90) {
                nlohmann::json food_items = {
                        {"ID",       std::to_string(id_dist(gen))},
                        {"name",     word},
                        {"quantity", 1},
                };
                jsonMann["food_items"].push_back(food_items);
            }
        }

        // After constructing the JSON Object dump it to the JSON string
        jsonStr = jsonMann.dump();

        // Initialize CURL for POST request
        CURL *curl;
        curl_global_init(CURL_GLOBAL_ALL);
        curl = curl_easy_init();

        // Confirm that CURL was initialized
        if (curl) {
            // Tell the Main thread to wait for the POST request to complete
            // (Used primarily to display success or failure)
            wait = true;

            // Specify the backend API access point
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8080/foodmanager/v1/add/");

            // Initialize the POST Request headers and specify JSON
            struct curl_slist *headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            // Add the header to the the POST request
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            // Specify that we only send one request
            curl_easy_setopt(curl, CURLOPT_POST, 1);

            // Add the JSON string to the POST Request and specify the length of the String
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, jsonStr.length());

            // Perform the POST Request
            CURLcode res = curl_easy_perform(curl);

            // If the POST request Failed:
            // Update the receipt contour to Red and output the error to terminal
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
                // If the POST request was successful Update the receipt contour in Green
                cv::Mat cleanFrame = frame.clone();
                cv::Scalar color(0, 255, 0);
                cv::drawContours(cleanFrame, std::vector<std::vector<cv::Point>>{largestTrapezoid}, 0, color, 20);
                std::string windowName = (mode == ReceiptOCR::DEBUG) ? "Receipt OCR - Debug Mode"
                                                                     : "Receipt OCR";
                cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
                cv::resizeWindow(windowName, 1080 * 0.5, 1920 * 0.5);
                cv::imshow(windowName, cleanFrame);
            }

            // Clean up the CURL
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
        // wait for half a second
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        // Inform the main thread to no longer wait
        wait = false;
    }

    /**
     * @brief Visualizes the scanning process.
     */
    void visualizeScanning() {
        // Check if we have detected a receipt, but has not completed scanning
        if (receiptDetected && !scannedReceipt) {

            // Initialize two empty images with the same width and height as the detected receipt
            // to visualize scanning the receipt
            cv::Mat scanningTrailFrame = cv::Mat::zeros(cv::Size(int(width), int(height)), CV_8UC3);
            scanningTrailFrame.setTo(cv::Scalar(0, 0, 0));
            cv::Mat tempFrame = scanningTrailFrame.clone();

            // Draw 5 lines on the image with decreasing opacity
            // The lines change direction and location based on the global direction and position variables
            for (int i = 0; i < 5; ++i) {
                cv::line(tempFrame, cv::Point(0, scanPosition - i * 50 * scanDirection),
                         cv::Point(frame.cols, scanPosition - i * 50 * scanDirection), cv::Scalar(0, 0, 255), 49);
                if (!tempFrame.empty()) cv::addWeighted(scanningTrailFrame, 1.0, tempFrame, 1 - 0.25 * i, 0.00, scanningTrailFrame);
            }

            // Initialize the Source points of the scanningTrailFrame for perspective warp.
            std::vector<cv::Point2f> srcPoints;
            srcPoints.emplace_back(0, 0);
            srcPoints.emplace_back(width, 0);
            srcPoints.emplace_back(0, height);
            srcPoints.emplace_back(width, height);

            // Get the perspective transformation matrix based on the previously extracted points and the constructed source points
            cv::Mat transformMatrix = cv::getPerspectiveTransform(srcPoints.data(), sortedPoints.data());

            // Initialize a Image to hold the perspective warped scanning images
            cv::Mat warpedScanningImage;
            cv::warpPerspective(scanningTrailFrame, warpedScanningImage, transformMatrix, frameClone.size());

            // Confirm that the warpedScanningImage can we overlaid onto the frameClone
            if (frameClone.size() != warpedScanningImage.size()) {
                std::cerr << "Error: Dimensions of 'frame' and 'warpedScanningImage' do not match."
                          << std::endl;
            } else if (frameClone.type() != warpedScanningImage.type()) {
                std::cerr << "Error: Type of 'frame' and 'warpedScanningImage' do not match." << std::endl;
                std::cerr << "Type of 'frame': " << frameClone.type() << ", Type of 'warpedScanningImage: "
                          << warpedScanningImage.type() << std::endl;
            } else {
                // Overlay the scanningImage on the frameClone
                cv::addWeighted(frameClone, 0.50, warpedScanningImage, 1.0, 0.0, frameClone);
            }

            // Update the position for the next time
            scanPosition += 200 * scanDirection;

            // if the position for the last line is outside the scanningImage flip the direction
            if (scanPosition >= frame.rows + 150 || scanPosition <= -150) {
                scanDirection *= -1;
            }
        }
    }

// Specify public class members (Visible outside the header file)
public:

    /**
     * @brief Parses command-line arguments and sets the operating mode.
     * @param argc Number of command-line arguments.
     * @param argv Array of command-line arguments.
     * @return 0 if successful, -1 if there is an error or help requested.
     */
    int getAppArguments(int argc, char *argv[]) {
        // Get the arguments sent to the applications:
        // If there are more than two arguments (Filename and optional command argument)
        // inform the user of the application usage
        if (argc > 2) {
            std::cerr << "Usage: " << argv[0] << " -[command] (Optional)" << std::endl;
            // Return error Value
            return -1;
        } else if (argc == 2) {
            // If the application received two arguments check if the secondary argument was debug to enable DEBUG Mode
            std::string command = argv[1];
            if (command == "-debug") {
                mode = ReceiptOCR::DEBUG;
                // Return Ok Value
                return 1;
            } else if (command == "-h") {
                // If the application received two arguments check if the secondary argument was help
                // to display the usage of the application
                std::cout << "Usage: " << argv[0] << " -[command] (Optional)\n" << std::endl;
                std::cout << argv[0] << ":\t Receipt OCR Scanner" << std::endl;
                std::cout << "\tEnter" << ":\t Scan Receipt\n" << std::endl;
                std::cout << "\tN/n" << ":\t New Scan" << std::endl;
                std::cout << "\tEsc" << ":\t Quit\n" << std::endl;
                std::cout << argv[0] << " -debug:\t Debug Mode - Check ComputerVision Output" << std::endl;
                // Return exit value
                return 0;
            } else {
                // If the application received two arguments,
                // and it is not recognized inform the user of the application usage
                std::cout << "Unknown Argument: " << argv[0] << " " << argv[1] << std::endl;
                std::cerr << "Usage: " << argv[0] << " -[command] (Optional)" << std::endl;
                std::cerr << "Help: " << argv[0] << " -h" << std::endl;
                // Return error value
                return -1;
            }
        }
        // If the application didn't receive additional arguments return Ok Value
        return 1;
    }

    /**
     * @brief Starts the OCR and scanning process.
     * @return 0 if successful, -1 if there is an error.
     */
    int start() {
        // Start Application by connecting to Camera
        cv::VideoCapture cap(0);

        // Check if the camera opened
        if (!cap.isOpened()) {
            // If the camera failed to open output error to terminal and return error Value
            std::cerr << "Error: Unable to open camera." << std::endl;
            return -1;
        }

        bool flag = true;       ///< Boolean used to stay in while loop (Implemented to have more control over when we stay in the loop)
        bool freeze = false;    ///< Boolean used to inform if the camera feed should be freezed
        bool scan = false;      ///< Boolean used to inform if We should start scanning (Only checked if receiptDetected is true)

        // Thread the OCR Extraction and POST Processes
        std::thread OCRPOSTThread([&]() {
            // Keep  the Thread alive while the flag is true
            while (flag) {
                // Check if we have detected a receipt but haven't scanned it
                if (receiptDetected && !scannedReceipt) {
                    // If we have detected a receipt we haven't scanned start the OCR process
                    getOCRSendPOST();
                    // After we have completed the OCR update that the receipt has been scanned
                    scannedReceipt = true;
                }
                // Sleep for one tenth of a second
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        // Keep the main thread alive while the flag is true
        while (flag) {
            // If we shouldn't freeze the frame, take the frame from the camera feed and rotate it 90 degrees
            // (This is due to phone connection having the orientation wrong)
            if (!freeze) {
                cap >> frame;
                cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            }

            // Confirm that we got a frame from the camera
            if (!frame.empty()) {

                // Check if a key was pressed and store the value
                int key = cv::waitKey(1);

                // If we pressed n we unfreeze and update the scanned receipt bool so we can scan a new receipt
                if (key == 'n' || key == 'N') {
                    freeze = false;
                    scannedReceipt = false;
                }

                // If we pressed 'Enter' we freeze the camera feed and start the scan
                // for safety we also set the scannedReceipt bool to false here manually
                if (key == 13) {
                    freeze = true;
                    scannedReceipt = false;
                    scan = true;
                }

                // Check if we have detected a receipt
                if (detectReceipt()) {
                    // Wait for 'Enter' press
                    if (scan) {
                        // Check if we got a receipt image from the scan
                        if (!receiptImage.empty()) {
                            // Stop scanning
                            scan = false;
                            // Update that we have detected a receipt so the OCR thread starts
                            receiptDetected = true;
                        } else {
                            receiptDetected = false;
                        }
                    }
                } else {
                    // While we haven't detected a board we make sure the bool is falsed
                    receiptDetected = false;
                }

                // Call the visualization ofor scanning function
                // if we have found a receipt and started scanning this will run
                visualizeScanning();

                // If the OCR thread hasn't told us to wait we display the camera feed
                if (!wait) {
                    std::string windowName = (mode == ReceiptOCR::DEBUG) ? "Receipt OCR - Debug Mode"
                                                                         : "Receipt OCR";
                    cv::namedWindow(windowName, cv::WINDOW_FREERATIO);
                    cv::resizeWindow(windowName, 1080 * 0.5, 1920 * 0.5);
                    // We double check that the frameClone contains data to avoid sigterm error
                    if (!frameClone.empty()) {
                        cv::imshow(windowName, frameClone);
                    }
                }

                // Wait for 'Esc key press
                if (key == 27) {
                    // set flag bool to false to exit loops
                    flag = false;
                }
            } else {
                // If we were unable to capture a frame from the camera we break the loop and exit the program
                std::cerr << "Error: Unable to capture frame." << std::endl;
                break;
            }
        }

        // Release the Camera and close all open windows
        // Join the OCR thread to the main thread for termination
        cap.release();
        cv::destroyAllWindows();
        OCRPOSTThread.join();

        // Return exit value
        return 0;
    }
};