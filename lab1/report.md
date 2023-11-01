# Lab Report

#### Author: *刘佳隆*

#### Student ID: *518010910009*

## Logistic Regression

### Parameter Settings

- Learning rate: 0.0001
- Iteration times: 100000 (In used)
- Convergence condition: 0.00001 (Not used)

### Result

![Logistic Regression](./lr_loss.png)
![Logistic Regression](./lr_accuracy.png)

- Testing accuracy: 0.915
- Training accuracy: 0.9475
- Training mean loss: 0.1755900108060871
- Training time: 64.88919115066528 seconds

## Support Vector Machine

### Parameter Settings

- Learning rate: 0.00005
- Lambda: 0.1
- Iteration times: 100000 (In used)
- Convergence condition: 0.00001 (Not used)

### Result

![Logistic Regression](./svm_loss.png)
![Logistic Regression](./svm_accuracy.png)

- Testing accuracy: 0.92
- Training accuracy: 0.9475
- Training mean loss: 0.14074722254232674
- Training time: 39.148269176483154 seconds

## Multi-layer Perceptron

### Parameter Settings

- Learning rate: 0.01
- Input layer size: 29
- Hidden layer 1 size: 20
- Hidden layer 2 size: 10
- Output layer size: 2
- Iteration times: 100000

### Result

![Logistic Regression](./mlp_loss.png)
![Logistic Regression](./mlp_accuracy.png)

- Testing accuracy: 0.885
- Training accuracy: 0.955
- Training mean loss: 0.3584943413734436
- Training time: 77.96472692489624 seconds