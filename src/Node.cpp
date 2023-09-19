#include "ANN/Node.hpp"

#include <iostream>
#include <random>
#include <cmath>

using namespace ANN;

void Node::clear()
{
    initVariables();
}

void Node::addParent(const Node::SharedPtr _parent)
{
    double weight = getRandWeight();

    parents_.push_back(std::make_pair(_parent, weight));
}

void Node::setInput(const double _input)
{
    input_ = _input;
}

double Node::getOutput()
{
    if (parents_.size() == 0 and not(std::isnan(input_)))
        return input_;

    if (std::isnan(output_))
    {
        double result = weight_0_;

        for (const auto &parentPair : parents_)
        {
            Node::SharedPtr parent = parentPair.first;
            double weight = parentPair.second;

            result += parent->getOutput() * weight;
        }

        output_ = activationFunc(result);
    }

    return output_;
}

void Node::add_error_term(const double _partial_error_term)
{
    if (std::isnan(error_term_))
        error_term_ = 0.0;

    error_term_ += _partial_error_term;
}

void Node::update_error_term()
{
    error_term_ = output_ * (1 - output_) * error_term_;

    // // Todo: Delete
    // {
    //     std::cout << error_term_ << " ";
    // }

    for (auto parentPair : parents_)
    {
        Node::SharedPtr parent = parentPair.first;
        double weight = parentPair.second;

        parent->add_error_term(weight * error_term_);
    }
}

void Node::update_weight(const double _learning_weight)
{
    weight_0_ += _learning_weight * error_term_;
    for (auto &parentPair : parents_)
    {
        Node::SharedPtr parent = parentPair.first;

        parentPair.second += _learning_weight * error_term_ * parent->getOutput();

        // // Todo: Delete
        // {
        //     std::cout << parentPair.second << " ";
        // }
    }

    initVariables();
}

void Node::initVariables()
{
    input_      = std::numeric_limits<double>::quiet_NaN();
    output_     = std::numeric_limits<double>::quiet_NaN();
    error_term_ = std::numeric_limits<double>::quiet_NaN();
}

double Node::getRandWeight()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> random_weight(0.0, 1.0);

    return random_weight(gen);
}

double Node::activationFunc(const double _input)
{
    double result = 0.0;

    if (not(std::isnan(input_)))
        return input_;

    switch (funcType_)
    {
    case FuncType::STEP:
    {
        result = _input > 0.0 ? 1.0 : -1.0;
        break;
    }

    case FuncType::SIGMOID:
    {
        result = 1 / (1 + std::exp(-1 * _input));
        break;
    }

    case FuncType::ReLU:
    {
        result = _input > 0.0 ? _input : 0.0;
        break;
    }

    default:
        std::cerr << "Invalid Activation Function Type" << std::endl;
        std::abort();
    }

    return result;
}

Node::Node(FuncType _funcType)
    : funcType_(_funcType)
{
    parents_.clear();

    initVariables();

    weight_0_ = getRandWeight();
}

Node::~Node()
{
    parents_.clear();
}