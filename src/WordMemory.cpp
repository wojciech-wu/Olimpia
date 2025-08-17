#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include '"WordsMemory.hpp

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
