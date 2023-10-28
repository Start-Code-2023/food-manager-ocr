#include "ReceiptOCR/receiptOCR.h"

int main(int argc, char *argv[]) {
    // Initialize receiptOCR
    ReceiptOCR receiptOcr;

    // Check that the application arguments if no errors start the OCR
    switch (receiptOcr.getAppArguments(argc, argv)) {
        case 0: return 0;
        case 1: return  receiptOcr.startOCR();
        case -1: return -1;
    }
}