
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

#include "gauss.hpp"


__global__ void applyMatToImg(uchar* extendImg, const double* convMatrix,
    const ulong convMatSize, const int extendImgCols, const int extendImgRows, 
    const int channels, const ulong shift) {

    const ulong x = threadIdx.x + blockDim.x * blockIdx.x;
    const ulong y = threadIdx.y + blockDim.y * blockIdx.y;

    
    if (y < extendImgRows && x < extendImgCols) {

        // Сonditions on x and y so that the kernel does not work 
        // with the edges of the matrix, but only with the image
        if (x >= shift && x < (extendImgCols - shift + 1)) {
            if (y >= shift && y < (extendImgRows - shift + 1)) {

                const ulong index = (y * extendImgCols + x) * channels;

                double sm[] = { 0.0, 0.0, 0.0 };

                const ulong x_shift = x - shift;
                const ulong y_shift = y - shift;

                // iterate over the entire convolution matrix
                for (ulong i = 0; i < convMatSize; ++i) {
                    for (ulong j = 0; j < convMatSize; ++j) {

                        for (int c = 0; c < channels; ++c)
                            sm[c] += convMatrix[i * convMatSize + j] * static_cast<double>(
                                extendImg[((y_shift+j) * extendImgCols + (x_shift+i)) * channels + c]
                                );
                    }
                }
                for (int c = 0; c < channels; ++c)
                    extendImg[index + c] = static_cast<uchar>(sm[c]);
            }
        }

    }

}


void extendingMat(cv::Mat &extendedMatrix, 
    const int &cols, const int &rows, const ulong &shift) {

    for (size_t i = shift; i < extendedMatrix.rows - shift; ++i) {
        for (size_t j = 0; j < shift; ++j) {
            extendedMatrix.at<cv::Vec3b>(i, j) = extendedMatrix.at<cv::Vec3b>(i, j + shift);
            extendedMatrix.at<cv::Vec3b>(i, j + cols + shift) = extendedMatrix.at<cv::Vec3b>(i, j + cols);
        }
    }

    for (size_t i = 0; i < extendedMatrix.cols; ++i) {
        for (size_t j = 0; j < shift; ++j) {
            extendedMatrix.at<cv::Vec3b>(j, i) = extendedMatrix.at<cv::Vec3b>(j + shift, i);
            extendedMatrix.at<cv::Vec3b>(j + shift + rows, i) = extendedMatrix.at<cv::Vec3b>(j + rows, i);
        }
    }
}

int main(int argc, char* argv[])
{

    if (argc != 5) {
        std::cout << "\nOne of the parameters was not entered\n";
        return 1;
    }

    // Obtain the necessary data to work with the image
    const std::string fileName = std::move(argv[1]);
    const std::string resFileName = std::move(argv[2]);
    const int sigma = std::stoul(argv[3]);
    const ulong matrixSize = std::stoul(argv[4]);
    const ulong shift = std::floor(matrixSize / 2);

    // Сreate a convolution matrix
    double* convMatrix = new double[matrixSize * matrixSize];
    cudaHostAlloc(&convMatrix, matrixSize * matrixSize * sizeof(double), cudaHostAllocDefault);
    CreateConvMatrix(convMatrix, sigma, matrixSize);

    
    cv::Mat img = cv::imread(fileName);

    // Сreate a magnified matrix to which to apply the convolution matrix
    cv::Mat extendedMatrix(img.rows + shift * 2, img.cols + shift * 2, img.type());

    size_t imgSize = extendedMatrix.cols * extendedMatrix.rows * img.channels();
    cudaHostAlloc(&extendedMatrix.data, imgSize * sizeof(uchar), cudaHostAllocDefault);

    // Сreate a region of a photograph that is applied 
    // to the expanded matrix to produce an image
    cv::Rect r(shift, shift, img.cols, img.rows);

    img.copyTo(extendedMatrix(r));

    // Fill the edges of the extended matrix by copying 
    // the edges of the photo to the edges of the matrix
    extendingMat(extendedMatrix, img.cols, img.rows, shift);

    uchar* extendImg_d = nullptr;
    double* convMatrix_d = nullptr;

    cudaMalloc(&extendImg_d, imgSize * sizeof(uchar));
    cudaMalloc(&convMatrix_d, matrixSize * matrixSize * sizeof(double));

    cudaMemcpy(extendImg_d, extendedMatrix.data, imgSize * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(convMatrix_d, convMatrix, matrixSize * matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8);
    dim3 gridDim(
        (extendedMatrix.cols + blockDim.x - 1) / blockDim.x,
        (extendedMatrix.rows + blockDim.y - 1) / blockDim.y
    );

    const int cols = extendedMatrix.cols;
    const int rows = extendedMatrix.rows;
    const int channels = extendedMatrix.channels();

    applyMatToImg<<<gridDim, blockDim >>>(extendImg_d, convMatrix_d, matrixSize, cols, rows, channels, shift);

    cudaMemcpy(extendedMatrix.data, extendImg_d, imgSize * sizeof(uchar), cudaMemcpyDeviceToHost);

    cv::imwrite(resFileName, extendedMatrix(r));

    cudaFreeHost(convMatrix);
    cudaFreeHost(extendedMatrix.data);
    cudaFree(extendImg_d);
    cudaFree(convMatrix);

    std::cout << "\nThe image is blurry!\n";
}
