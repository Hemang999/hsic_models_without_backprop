# The HSIC Bottleneck: Deep Learning Without Back-Propagation in Keras
This is my attempt at replicating and implementing the HSIC bottleneck models that work without the traditional backpropagation as mentioned in the paper "The HSIC Bottleneck: Deep Learning without Back-Propagation."

https://arxiv.org/abs/1908.01580

The authors introduced the HSIC (Hilbert-Schmidt independence criterion) bottleneck for training deep neural networks. The HSIC bottleneck is an alternative to the conventional cross-entropy loss and backpropagation that has a number of distinct advantages. It mitigates exploding and vanishing gradients, resulting in the ability to learn very deep networks without skip connections. There is no requirement for symmetric feedback or update locking. These models' performance on MNIST/FashionMNIST/CIFAR10 classification is comparable to backpropagation with a cross-entropy target, even when the system is not encouraged to make the output resemble the classification labels. Appending a single layer trained with SGD (without backpropagation) to reformat the information further improves performance.

I have attempted to implement the algorithm in Keras instead of PyTorch as the authors of the paper did. Keras, and by extension Tensorflow, automatically apply backpropagation to the models. However, I have attempted to fix this in my implementation.
