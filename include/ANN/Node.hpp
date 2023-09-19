#pragma once

#include <vector>
#include <memory>

namespace ANN
{
    class Node : public std::enable_shared_from_this<Node>
    {
    public:
        using SharedPtr = std::shared_ptr<Node>;

        enum FuncType{STEP, SIGMOID, ReLU};
    
    public:
        void clear();
        void addParent(const Node::SharedPtr _parent);
        void setInput(const double _input);
        double getOutput();

        void add_error_term(const double _partial_error_term);
        void update_error_term();

        void update_weight(const double _learning_rate);

    private:
        void initVariables();

        double getRandWeight();
        double activationFunc(const double _input);

    private:
        std::vector<std::pair<Node::SharedPtr, double>> parents_;
        double weight_0_;

        FuncType funcType_;

        double input_;
        double output_;

        double error_term_;

    public:
        Node(
            FuncType _funcType = FuncType::SIGMOID);
        ~Node();
    }; // class Node
} // namespace ANN