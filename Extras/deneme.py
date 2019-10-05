import pandas as pd

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
        counter += 1

        print(output, rate, weight1, weight2, bias, counter)

        flag = not ((correct_output - output) == 0)
        
        # is_correct_string = 'Yes' if output == correct_output else 'No'
        # outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

print(weight1, weight2, bias, counter)

# # Print output
# num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
# output_frame = pd.DataFrame(outputs, columns=['Input-1', '  Input-2', '  Linear Combination', '  Activation Output', '  Is Correct'])
# if not num_wrong:
#     print('Nice!  You got it all correct.\n')
# else:
#     print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
# print(output_frame.to_string(index=False))
