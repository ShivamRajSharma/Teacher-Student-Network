# Teacher-Student-Network

Pytorch implementation of Teacher-Student Network which idea of model compression by teaching a smaller network, step by step, exactly what to do using a bigger already trained network. While large teacher model have higher knowledge capacity than small student model, this capacity might not be fully utilized. It can be computationally just as expensive to evaluate a model even if it utilizes little of its knowledge capacity. Knowledge distillation transfers knowledge from a large model to a smaller model without loss of validity. Teacher supervising the student's learning helps in achieving a better score metric than it was able to achieve when the student network tried learning in a supervised way.


<p align="center">
  <img src="https://miro.medium.com/max/936/1*8KqNtABnNXM527JK9UuBUQ.jpeg" height="280" />
</p>


## Dataset Information 
The Student and teacher were trained on CIFAR100 and CIFAR10 dataset. The dataset contains 50k train images containing 100, 10 classes respectively 

## Usage

1) Install all the libraries required ```pip install -r requirements``` .
2) Run ```python3 train.py``` .
3) For inference run ```python3 predict.py``` .


## Results 
1) CIFAR 100

| Method  | Accuracy |
| :---: | :---: |
| Teacher Model | 54.3% |
| Student Model | 28.3% |
| Teacher-Student Model | 33.1% |


