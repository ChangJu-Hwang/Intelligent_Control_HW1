#include "ANN/logger.hpp"

#include <iostream>
#include <cmath>

#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h> // Need to install libyaml-cpp-dev

using namespace ANN;

void Logger::record_SumOfSquaredErrors(
    const int _outputIdx, const Data &_output, const Data &_ANN_output)
{
    if (not(sum_of_squared_errors_.contains(_outputIdx)))
    {
        std::vector<double> errLog;
        errLog.clear();

        sum_of_squared_errors_.insert(
            std::make_pair(_outputIdx, errLog));
    }

    sum_of_squared_errors_[_outputIdx].push_back(
        calculate_SumOfSquaredErrors(_output, _ANN_output));
}

void Logger::export_SumOfSquaredErrors()
{
    std::filesystem::remove_all("../log");
    std::filesystem::create_directories("../log");

    YAML::Node node;
    for (const auto &errorPair : sum_of_squared_errors_)
    {
        auto outputIdx = errorPair.first;
        auto errors = errorPair.second;

        node[std::to_string(outputIdx)] = errors;
    }

    std::string fPath = "../log/log.yaml";
    std::ofstream fout(fPath);
    fout << node;
}

double Logger::calculate_SumOfSquaredErrors(
    const Data &_output, const Data &_ANN_output)
{
    double SumOfSquaredErrors = 0.0;

    if (_output.size() != _ANN_output.size())
        std::abort();

    for (int idx = 0; idx < _output.size(); idx++)
        SumOfSquaredErrors += 0.5*std::pow(_output[idx] - _ANN_output[idx], 2);

    return SumOfSquaredErrors;
}

Logger::Logger()
{
    std::cout << "Logger has been initialized" << std::endl;
}

Logger::~Logger()
{
    std::cout << "Logger has been terminated" << std::endl;
}