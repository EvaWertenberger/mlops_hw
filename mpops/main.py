import utils


# Test the division function
print(utils.div_numbers(10, 2))  # Output: 5

try:
    print(utils.div_numbers(10, 0))
except ValueError as e:
    print(e)  # Output: b cannot be zero

# Test the summation function
print(utils.summ_numbers(10, 2))  # Output: 12