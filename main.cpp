#include "ReceiptOCR/receiptOCR.h"

int main(int argc, char *argv[]) {
    // Initialize receiptOCR
    ReceiptOCR receiptOcr;

    // Check that the application arguments if no errors start the OCR
    switch (receiptOcr.getAppArguments(argc, argv)) {
        // If we received exit value we exit the application
        case 0: return 0;
        // if we received no error from the app arguments
        // we start the application
        // based on the return values from the start function we can exit with error or exit value,
        case 1: return receiptOcr.start();
        // If we received error value we exit with error
        case -1: return -1;
    }
}