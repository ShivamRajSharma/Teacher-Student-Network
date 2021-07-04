# Teacher-Student-Network

Pytorch implementation of Teacher-Student Network which idea of model compression by teaching a smaller network, step by step, exactly what to do using a bigger already trained network. While large teacher model have higher knowledge capacity than small student model, this capacity might not be fully utilized. It can be computationally just as expensive to evaluate a model even if it utilizes little of its knowledge capacity. Knowledge distillation transfers knowledge from a large model to a smaller model without loss of validity. Teacher supervising the student's learning helps in achieving a better score metric than it was able to achieve when the student network tried learning in a supervised way.



## Usage

1) Install all the libraries required ```pip install -r requirements``` .
2) Download the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place it inside the ```input/``` .
3) Run ```python3 train.py``` .
4) For inference run ```python3 predict.py``` .
