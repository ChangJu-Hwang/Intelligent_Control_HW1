#pragma once

#include <vector>
#include <unordered_map>

namespace ANN
{
    using Data = std::vector<double>;

    class Logger
    {
    public:
        void record_SumOfSquaredErrors(
            const int _outputIdx, const Data &_output, const Data &_ANN_output);

        void export_SumOfSquaredErrors();

    private:
        double calculate_SumOfSquaredErrors(
            const Data &_output, const Data &_ANN_output);

    private:
        std::unordered_map<int, std::vector<double>> sum_of_squared_errors_;

    public:
        Logger();
        ~Logger();
    }; // class Logger
} // namespace ANN