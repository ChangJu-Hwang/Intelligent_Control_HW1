#include "ANN/Network.hpp"

#include <iostream>

#include <yaml-cpp/yaml.h> // Need to install libyaml-cpp-dev

using namespace ANN;

void Network::run()
{
    std::cout << "ANN::Network::run()" << std::endl;

    for (int epoch = 0; epoch < epochs_; epoch++)
    {
        for (int iteration = 0; iteration < dataSet_.size(); iteration++)
            learn_once(iteration);
    }

    for (int dataIdx = 0; dataIdx < dataSet_.size(); dataIdx++)
    {
        compareInputOutput(dataIdx);
        std::cout << "============================================" << std::endl;
    }
}

void Network::initVariables()
{
    network_.clear();
    dataSet_.clear();

    dataFileDir_ = "../data/" + dataFileName_ + ".yaml";
    paramFileDir_ = "../param/" + paramFileName_ + ".yaml";

    hidden_layer_num_ = 0;
    hidden_layer_size_ = 0;

    learning_rate_ = 0.0;
    epochs_ = 0;
}

void Network::importSettings()
{
    YAML::Node params = YAML::LoadFile(paramFileDir_);

    for (const auto &param : params["hidden_layer"])
    {
        hidden_layer_num_ = param["num"].as<int>();
        hidden_layer_size_ = param["size"].as<int>();
    }

    learning_rate_ = params["learning_rate"].as<double>();
    epochs_ = params["epochs"].as<int>();
}

void Network::importDataSet()
{
    YAML::Node dataSet = YAML::LoadFile(dataFileDir_);

    for (const auto &data : dataSet["data_set"])
    {
        auto components = data["data"];

        for (const auto &component : components)
        {
            auto input = component["input"].as<std::vector<double>>();
            auto output = component["output"].as<std::vector<double>>();

            if (dataSet_.size() == 0 or
                (dataSet_.size() > 0 and
                 dataSet_[0].first.size() == input.size() and
                 dataSet_[0].second.size() == output.size()))
            {
                dataSet_.push_back(std::make_pair(input, output));
            }
        }
    }
}

void Network::genearteLayers()
{
    generateLayer(dataSet_[0].first.size(), Node::FuncType::SIGMOID);

    for (int i = 0; i < hidden_layer_num_; i++)
        generateLayer(hidden_layer_size_, Node::FuncType::SIGMOID);

    generateLayer(dataSet_[0].second.size(), Node::FuncType::SIGMOID);
}

void Network::generateLayer(const int _size, Node::FuncType _funcType)
{
    Layer layer;

    layer.clear();
    for (int i = 0; i < _size; i++)
    {
        auto nodePtr = std::make_shared<Node>(_funcType);
        layer.push_back(nodePtr);
    }

    network_.push_back(layer);
}

void Network::connectLayer()
{
    for (auto layerIter = network_.begin(); layerIter != network_.end(); ++layerIter)
    {
        if (layerIter == network_.begin())
            continue;

        auto parentLayerIter = std::prev(layerIter, 1);

        for (auto nodeIter = (*layerIter).begin(); nodeIter != (*layerIter).end(); ++nodeIter)
        {
            for (auto parentNodeIter = (*parentLayerIter).begin(); parentNodeIter != (*parentLayerIter).end(); ++parentNodeIter)
                nodeIter->get()->addParent(*parentNodeIter.base());
        }
    }
}

void Network::learn_once(const int _dataIdx)
{
    front_propagation(_dataIdx);
    update_error_terms(_dataIdx);
    update_weights();

    // // Todo: Delete
    // {
    //     std::cout << std::endl;
    // }
}

void Network::front_propagation(const int _dataIdx)
{
    setInputs(_dataIdx);

    for (const auto &output_node : network_.back())
        output_node->getOutput();

    // // Todo: Delete
    // {
    //     std::cout << "Input : ";
    //     for (const auto inputNode : network_[0])
    //     {
    //         std::cout << inputNode->getOutput() << " ";
    //     }
    //     std::cout << std::endl;

    //     std::cout << "Hidden: ";
    //     for (const auto hiddenNode : network_[1])
    //     {
    //         std::cout << hiddenNode->getOutput() << " ";
    //     }
    //     std::cout << std::endl;

    //     std::cout << "Output: ";
    //     for (const auto outputNode : network_.back())
    //     {
    //         std::cout << outputNode->getOutput() << " ";
    //     }
    //     std::cout << std::endl;
    // }
}

void Network::setInputs(const int _dataIdx)
{
    if (dataSet_[0].first.size() != network_[0].size())
    {
        std::cerr << "Invalide node numbers in Input Layer" << std::endl;
        std::abort();
    }

    for (int inputIdx = 0; inputIdx < dataSet_[_dataIdx].first.size(); inputIdx++)
    {
        double input = dataSet_[_dataIdx].first.at(inputIdx);
        network_[0][inputIdx]->setInput(input);
    }
}

void Network::update_error_terms(const int _dataIdx)
{
    // // Todo: Delete
    // {
    //     std::cout << "Error: ";
    // }

    for (int nodeIdx = 0; nodeIdx < network_.back().size(); nodeIdx++)
    {
        Node::SharedPtr outputNode = network_.back().at(nodeIdx);

        double target_output = dataSet_[_dataIdx].second.at(nodeIdx);
        double node_output = outputNode->getOutput();

        outputNode->add_error_term(target_output - node_output);

        // // Todo: Delete
        // {
        //     std::cout << target_output - node_output << " ";
        // }
    }
    // // Todo: Delete
    // {
    //     std::cout << std::endl;
    // }

    for (auto layerIter = network_.rbegin(); layerIter != (network_.rend() - 1); ++layerIter)
    {
        // // Todo: Delete
        // {
        //     std::cout << "Error_Term: ";
        // }
        for (auto &node : *(layerIter))
            node->update_error_term();
        // // Todo: Delete
        // {
        //     std::cout << std::endl;
        // }
    }
}

void Network::update_weights()
{
    for (auto layerIter = network_.rbegin(); layerIter != (network_.rend() - 1); ++layerIter)
    {
        // // Todo: Delete
        // {
        //     std::cout << "Weight:";
        // }
        for (auto &node : *(layerIter))
        {
            // // Todo: Delete
            // {
            //     std::cout << std::endl;
            // }
            node->update_weight(learning_rate_);
        }
        // // Todo: Delete
        // {
        //     std::cout << std::endl;
        // }
    }
}

void Network::CacheData_clear()
{
    for (auto &layer : network_)
    {
        for (auto &node : layer)
            node->clear();
    }
}

void Network::compareInputOutput(const int _dataIdx)
{
    CacheData_clear();

    front_propagation(_dataIdx);

    std::cout << "Input : ";
    for (const auto value : dataSet_[_dataIdx].first)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (const auto outputNode : network_.back())
    {
        std::cout << outputNode->getOutput() << " ";
    }
    std::cout << std::endl;
}

Network::Network(
    const std::string _dataFileName,
    const std::string _paramFileName)
    : dataFileName_(_dataFileName), paramFileName_(_paramFileName)
{
    initVariables();

    importSettings();
    importDataSet();

    genearteLayers();
    connectLayer();

    std::cout << "ANN::Network has been initialized" << std::endl;
}

Network::~Network()
{
    std::cout << "ANN::Network has been terminated" << std::endl;
}