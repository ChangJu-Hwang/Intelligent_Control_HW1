# Intelligent_Control_HW1
Homework for 2023-2 Intelligent Control.

Details: Implement Artificial Neural Network.

## Dependencies
The code requires the external library [yaml-cpp](https://github.com/jbeder/yaml-cpp). If you are using Ubuntu, you can install it simply by
```shell script
sudo apt-get install libyaml-cpp-dev 
```
## Parameters
 | Parameter             | Type   | Definition                                              |
 | ---------------       | ------ | --------------------------------------------------------|
 | hidden_layer.num      | int    | Number of hidden layers                                 |
 | hidden_layer.size     | int    | Number of nodes(neuron) in hidden layer                 |
 | learning_rate         | double | Learning rate for ANN                                   |
 | epochs                | int    | One entire passing of training data through the ANN     |

## How to Use
#### 1. Navigate into the source directory, create build folder and run `CMake`:

```shell script
mkdir build
cd build
cmake .. &&make
```
#### 2. Run the code
```shell script
./ANN
```