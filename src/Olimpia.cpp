// Olimpia AI
// Created by Wojciech Wasilewski
// (c) 2025 — All rights reserved
// License: MIT
// neural_network.cpp
// Core logic for feedforward neural network with backpropagation

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <fstream>
#include <cctype>
#include <algorithm>
#include "WordsMemory.hpp"
#include "SimpleNN.hpp"

void trainModel(SimpleNN& model, const std::vector<double>& input, const std::vector<double>& target, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        model.train(input, target);
    }
    std::cout << "Training complete. Olimpia AI is ready.\n";
}

void storeWordsInMemory(const std::string& line) {
    std::istringstream iss(line);
    std::string word;

    while (iss >> word) {
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        if (!word.empty() && std::isalpha(word[0])) {
            char initial = word[0];
            std::string filename = std::string(1, initial) + ".txt";

            std::ofstream file(filename, std::ios::app);
            if (file.is_open()) {
                file << word << "\n";
                file.close();
            }
        }
    }
}

void runInteractiveLoop(SimpleNN& model) {
    std::string line;

    while (true) {
        std::cout << "\nEnter 3 input values separated by spaces (or type 'exit' to quit): ";
        std::getline(std::cin, line);

        if (line == "exit") {
            std::cout << "Goodbye from Olimpia AI.\n";
            break;
        }

        if (line == "show memory") {
            for (char c = 'a'; c <= 'z'; ++c) {
                std::string filename = std::string(1, c) + ".txt";
                std::ifstream file(filename);
                if (file.is_open()) {
                    std::string word;
                    std::cout << c << ": ";
                    while (std::getline(file, word)) {
                        std::cout << word << " ";
                    }
                    std::cout << "\n";
                    file.close();
                }
            }
            continue;
        }

        // Store all words in memory
        storeWordsInMemory(line);

        // Try to parse numeric input
        std::istringstream iss(line);
        std::vector<double> userInput;
        std::string token;
        bool valid = true;

        while (iss >> token) {
            try {
                double val = std::stod(token);
                userInput.push_back(val);
            }
            catch (...) {
                valid = false;
                break;
            }
        }

        if (valid && userInput.size() == 3) {
            std::vector<double> output = model.forward(userInput);
            std::cout << "Olimpia AI output: ";
            for (double o : output)
                std::cout << o << " ";
            std::cout << std::endl;
        }
        else {
            std::cout << "Text input received and stored. Olimpia is listening.\n";
        }
    }

    std::cout << "Press Enter to close...\n";
    std::cin.get();
}

int main() {
    std::cout << "=================================================================================================================\n";
    std::cout << "                         Olimpia AI\n";
    std::cout << "                                       Created by Wojciech Wasilewski\n";
    std::cout << "                                       (c) 2025 — All rights reserved\n";
    std::cout << "                                       Lightweight neural network\n";
    std::cout << "    Version: 1.0.0        ssh-keygen / ssh key timestamp : Aug 17, 2025\n";
    std::cout << "=================================================================================================================\n\n";

    SimpleNN model(3, 4, 2, 0.1);

    std::vector<double> input = { 0.5, 0.2, 0.8 };
    std::vector<double> target = { 0.1, 0.9 };

    trainModel(model, input, target, 1000);
    runInteractiveLoop(model);

    std::cout << "\nPress Enter to exit...\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return 0;
}
