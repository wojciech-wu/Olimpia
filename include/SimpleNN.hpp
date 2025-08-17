// Olimpia AI
// Created by Wojciech Wasilewski
// (c) 2025 — All rights reserved
// License: MIT

#pragma once
#include <vector>

#ifdef OLIMPIACORE_EXPORTS
#define OLIMPIACORE_API __declspec(dllexport)
#else
#define OLIMPIACORE_API __declspec(dllimport)
#endif

class OLIMPIACORE_API SimpleNN {
public:
    // Constructor and destructor
    SimpleNN(int inputSize, int hiddenSize, int outputSize, double learningRate);
    ~SimpleNN();

    // Public interface
    std::vector<double> forward(const std::vector<double>& input);
    void train(const std::vector<double>& input, const std::vector<double>& target);

private:
    // Network architecture
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;

    // Weights
    std::vector<std::vector<double>> weights1;
    std::vector<std::vector<double>> weights2;

    // Internal helpers
    std::vector<double> sigmoid(const std::vector<double>& x) const;
    std::vector<double> dot(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) const;
};
