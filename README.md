# Neural Network Model Evolution

This document outlines the evolution of three neural network models, detailing the changes made in each iteration to achieve the final desired performance.

## Model 1: Initial Approach

### Target
*   Create a basic neural network with a parameter count around 8k.
*   Achieve the best possible accuracy.

### Results
*   **Parameters**: 10.4k
*   **Best Train Accuracy**: 98.55%
*   **Best Test Accuracy**: 98.43%

### Analysis
The initial model was a good starting point, but it exhibited overfitting in the final layers. The parameter count was also slightly higher than the target. The next steps were to introduce techniques to combat overfitting and reduce the number of parameters.

## Model 2: Improving the Architecture

### Target
*   Improve the architecture by adding:
    *   Batch-norm layers
    *   Dropout layers
    *   A Global Average Pooling (GAP) layer
*   Reduce the parameter count to less than 10k.
*   Reduce overfitting.

### Results
*   **Parameters**: 8.0k
*   **Best Train Accuracy**: 98.86%
*   **Best Test Accuracy**: 99.2%

### Analysis
- **Batch-norm**: Adding batch-norm alone resulted in overfitting.
- **Dropout**: Dropout helped in reducing the gap between training and testing accuracy, but overfitting persisted.
- **GAP Layer**: The GAP layer successfully reduced the parameter count to well below the 10k target, but at the cost of a slight drop in accuracy.
- **Increased Capacity**: To counteract the accuracy drop, the model's capacity was increased, which pushed the test accuracy up to 99.2%.

While the model was much improved, it still hadn't reached the desired 99.4% accuracy. The analysis suggested that the next steps should involve exploring the optimal placement of max pooling, using image augmentation, and implementing learning rate schedulers.

## Model 3: Fine-tuning and Final Result

### Target
*   Achieve a test accuracy of 99.4% consistently.
*   Keep the parameter count under 8k.
*   Implement:
    *   Correct placement of max pooling
    *   Image augmentation
    *   Learning rate schedulers

### Results
*   **Parameters**: 7.1k
*   **Best Train Accuracy**: 99.01%
*   **Best Test Accuracy**: 99.43%

### Analysis
- **Max Pooling**: Placing the max pooling layer after a receptive field of 5x5 (after two convolution layers) improved accuracy, but the model started to underfit.
- **Image Augmentation**: To combat underfitting, image augmentation techniques were introduced to increase the effective size of the training dataset. This led to a significant improvement in accuracy.
- **Additional Augmentation**: Further augmentation, specifically `RandomAffine` transformations, pushed the test accuracy to 99.34%.
- **Scheduler**: The final piece of the puzzle was adding a `StepLR` learning rate scheduler. This allowed the model to fine-tune its weights more effectively in the later stages of training, finally achieving the target test accuracy of 99.43%.

## Summary

This project demonstrates a systematic approach to improving a neural network model. Starting with a basic architecture, we iteratively introduced new techniques, including batch normalization, dropout, global average pooling, thoughtful layer placement, image augmentation, and learning rate scheduling. Each change was evaluated, and the insights gained from each step informed the next. The final model, with only 7.1k parameters, successfully achieved the ambitious goal of 99.43% test accuracy, showcasing the power of this iterative and analytical approach to model development.