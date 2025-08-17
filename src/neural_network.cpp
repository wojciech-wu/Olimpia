// Olimpia AI
// Created by Wojciech Wasilewski
// (c) 2025 — All rights reserved
// License: MIT
// Core logic for feedforward neural network with backpropagation

#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "WordsMemory.hpp"
#include "SimpleNN.hpp"
SimpleNN::SimpleNN(int inputSize, int hiddenSize, int outputSize, double learningRate)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(learningRate) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    weights1.resize(hiddenSize, std::vector<double>(inputSize));
    weights2.resize(outputSize, std::vector<double>(hiddenSize));

    for (auto& row : weights1)
        for (auto& val : row)
            val = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;

    for (auto& row : weights2)
        for (auto& val : row)
            val = ((double)std::rand() / RAND_MAX) * 2.0 - 1.0;
}

SimpleNN::~SimpleNN() {
    // No dynamic memory to release, but required for proper DLL linkage
}

std::vector<double> SimpleNN::sigmoid(const std::vector<double>& x) const {
    std::vector<double> result;
    result.reserve(x.size());
    for (double val : x)
        result.push_back(1.0 / (1.0 + std::exp(-val)));
    return result;
}

std::vector<double> SimpleNN::dot(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) const {
    std::vector<double> result(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i)
        for (size_t j = 0; j < vec.size(); ++j)
            result[i] += mat[i][j] * vec[j];
    return result;
}

std::vector<double> SimpleNN::forward(const std::vector<double>& input) {
    std::vector<double> hidden = sigmoid(dot(weights1, input));
    return sigmoid(dot(weights2, hidden));
}

void SimpleNN::train(const std::vector<double>& input, const std::vector<double>& target) {
    std::vector<double> hidden = sigmoid(dot(weights1, input));
    std::vector<double> output = sigmoid(dot(weights2, hidden));

    std::vector<double> outputError(outputSize);
    for (int i = 0; i < outputSize; ++i)
        outputError[i] = (target[i] - output[i]) * output[i] * (1.0 - output[i]);

    std::vector<double> hiddenError(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i)
        for (int j = 0; j < outputSize; ++j)
            hiddenError[i] += outputError[j] * weights2[j][i] * hidden[i] * (1.0 - hidden[i]);

    for (int i = 0; i < outputSize; ++i)
        for (int j = 0; j < hiddenSize; ++j)
            weights2[i][j] += learningRate * outputError[i] * hidden[j];

    for (int i = 0; i < hiddenSize; ++i)
        for (int j = 0; j < inputSize; ++j)
            weights1[i][j] += learningRate * hiddenError[i] * input[j];
}
