import pandas as pd

# -- CODE SOLVE --
# For the second example, where the line is described by 3x1 + 4x2 - 10 = 0, 
# if the learning rate was set to 0.1, 
# how many times would you have to apply the perceptron trick 
# to move the line to a position where the blue point (DESIRED_OUTPUT = 1, BLUE_POINT[1,1] = 0), 
# at (1, 1), is correctly classified?

# 0.2 = 0.20000000000000001
# That's why we must use Decimal
# Otherwise the result will be different
from decimal import Decimal

# Weights and Bias
weight1 = Decimal('3.0')
weight2 = Decimal('4.0')
bias = Decimal('-10.0')

# Learning Rate
LEARNING_RATE = Decimal('0.1')

# Inputs and outputs
input_bias = 1
test_inputs = [(1, 1)]
correct_outputs = [1]
outputs = []

# Flag
flag = True

# Counter of Steps
counter = 0

# Generate and check output
while (flag):
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= 0)    
        rate = LEARNING_RATE * (correct_output - output)
        weight1 += test_input[0] * rate
        weight2 += test_input[1] * rate
        bias += input_bias * rate
        
        flag = not ((correct_output - output) == 0)
        if(flag):
            counter += 1

print(counter)
