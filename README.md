###  Task Description

In this task, we aim to generate a two-dimensional dataset where the decision boundary between two classes (A and B) is defined by a non-linear curve of the form:

`$$y = ax^2 + x$$`

Points lying on one side of this curve will be labeled as class A, while points on the other side will be labeled as class B.

We will investigate how effectively **logistic regression** can classify this data as the parameter \( a \) is varied, and compare its performance with that of a **small neural network**. The analysis will also explore how both models respond to changes in:

- The number of data points  
- The class balance between A and B  
- The size (capacity) of the neural network  

Our goal is to understand the limitations of linear models like logistic regression in handling non-linear boundaries, and to evaluate how neural networks adapt to such variations in data complexity.
