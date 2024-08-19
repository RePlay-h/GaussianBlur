#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <limits>

constexpr double pi = 3.141592653589793;

typedef std::vector<std::vector<double>> ConvolutionMatrix;
typedef unsigned long ulong;

static double GaussianFunc(const double& x, const double& y, const double& sigma) {

    return (1.0 / (2 * pi * sigma * sigma))
        / std::exp(((x * x) + (y * y)) / (2 * sigma * sigma));

}

void CreateConvMatrix(double *convMatrix, const int& sigma, const size_t& matrixSize) {

    const int shift = std::floor(matrixSize / 2);

    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {

            const int x = i - shift;
            const int y = j - shift;
            convMatrix[i + j * matrixSize] = GaussianFunc(x, y, sigma);

        }
    }
}