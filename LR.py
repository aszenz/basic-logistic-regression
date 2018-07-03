import math
import numpy as np
class LogisticRegression:
  """
  Logistic Regression for Classification
  """

  def __init__(self, no_of_features=2):
    """Initialize parameters to zeros"""
    # no of parameters will be equal to no of features
    self.parameters = np.zeros((no_of_features), dtype='float')
    self.no_of_features = no_of_features

  def predict(self, input_data):
    """Predict output given input"""
    # return the output
    return 1 / ( 1 + math.e ** -np.dot(self.parameters.transpose(), input_data) )
  
  def hybrid_gd_fit(self, inputs, outputs, learning_rate=0.01, iterations=1000, batch_size=1):
    """Train the model as ordinary least squares regression model with custom batch size"""
    no_of_examples = len(inputs)
    # custom batch size algorithm
    while iterations > 0:
      for i in range(0, no_of_examples, batch_size):
        end = no_of_examples if i + batch_size > no_of_examples else i + batch_size
        predictions = np.array( [self.predict(j) for j in inputs[i:end] ])
        # update each parameter (weight) based on the derivative for a batch(batch_size) of training examples simultaneously
        for index in range(len(self.parameters)):
          derivative = 0
          for k in range(batch_size):
            if i + k >= no_of_examples:
               break
            else:
              derivative += (outputs[i+k] - predictions[k]) * inputs[i+k][index] 
          # LMS Rule -- Least Mean Square Update Rule
          self.parameters[index] = self.parameters[index] + learning_rate * derivative
      iterations -= 1
    print('Model parameters: %s' % self.parameters)

inputs = [[2], [-3], [-1], [-4], [6], [13], [12], [1], [22], [61], [-14], [-19], [-22], [-29]]
outputs = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
inputs = np.array(inputs)
outputs = np.array(outputs)
LR = LogisticRegression(no_of_features=1)
LR.hybrid_gd_fit(inputs, outputs)
print('Input: %f ; Output: %f' % (7, LR.predict(np.array([7]))))
print('Input: %f ; Output: %f' % (-8, LR.predict(np.array([-8]))))
