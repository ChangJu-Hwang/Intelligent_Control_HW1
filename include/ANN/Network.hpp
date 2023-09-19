#pragma once

#include <vector>

#include "ANN/Node.hpp"

namespace ANN
{
    using Layer = std::vector<ANN::Node::SharedPtr>;
    using Data = std::vector<double>;

    class Network
    {
    private:
    public:
        void run();

    private:
        // Initial Setting
        void initVariables();
        void importSettings();
        void importDataSet();

        // Genearte network
        void genearteLayers();
        void generateLayer(
            const int _size, Node::FuncType _funcType);
        void connectLayer();

        // Learning
        void learn_once(const int _dataIdx);

        // Forward Propagation
        void front_propagation(const int _dataIdx);
        void setInputs(const int _dataIdx);

        // Backward Propagation
        void update_error_terms(const int _dataIdx);
        void update_weights();

        // Util
        void CacheData_clear();
        void compareInputOutput(const int _dataIdx);

    private:
        std::vector<Layer> network_;

        std::string dataFileName_;
        std::string dataFileDir_;

        std::string paramFileName_;
        std::string paramFileDir_;

        int hidden_layer_num_;
        int hidden_layer_size_;
        double learning_rate_;
        int epochs_;

        std::vector<std::pair<Data, Data>> dataSet_;

    public:
        Network(
            const std::string _dataFileName = std::string("demo"),
            const std::string _paramFileName = std::string("ANN"));
        ~Network();
    }; // class Network
} // namespace ANN