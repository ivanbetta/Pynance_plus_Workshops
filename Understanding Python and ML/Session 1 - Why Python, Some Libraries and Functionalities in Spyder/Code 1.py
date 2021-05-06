
# Understanding Python

'The objective of this code is understand basic functionalities in Python'

# Libraries

'Always add libraries on the top and comments to your codes'

import math as mt
'Basic mathematical functions'

# Coding Basic Tips

'Rememeber always to check 4 spaces for indentation'
example = 1
for i in range(1,10):
    example = example + 1
    print(example)

'To jump the code to the next line just add \
    '
sum = 1 + \
    2 + \
    3
print(sum)

# Basic operations

x = 5
y = 26

plus = x + y
print(plus)
minus = x - y
print(minus)
product = x * y
print(product)
div = x / y
print(div)

# Variables Class

integer = 9
floating = 3.0
string = 'python'

print(integer)
print(floating)
print(string)

print(type(string))

# Data types

text_type = 'text'
print(text_type)

numeric_type = mt.pi
print(numeric_type)

set_type = {5,2,3,1,4}
print(set_type)

# Objects 

all_data_in_python_is_an_object = 100

print(id(all_data_in_python_is_an_object))
'The identity never changes and is the address in memory'

hundred = 100
print(id(hundred))

print(type(all_data_in_python_is_an_object))
'type cannot change and specifies operations allowed + values that can hold'

'Mutable ojects can change'
list_example =[1,2,3,4]

print(list_example)
print(list_example[2])
list_example[2] = 50
print(list_example)

dictionary_example = {"brand": "Ford",
                      "model": "Mustang",
                      "year": 1964}

print(dictionary_example)
print(dictionary_example["brand"])

'Immutable ojects'
tuple_example =(1,2,3,4)

print(tuple_example)
print(tuple_example[2])
# tuple_example[2] = 50
# print(tuple_example)

# If's

print('Please write your age:')
age = input()
print ('Your age is ' + age)

if int(age) >= 18:
    print('You can drink beer! Congrats!')
else:
    print('Please order soda.')

# Loops

i = 1
while i <10:
    i = i +1
    print(i)

for j in range(4):
    print(j)

# Functions

def function_test():
    print('function test READY')

function_test()

# Libraries

math_function = mt.sqrt(36)
print(math_function)

# References

'https://www.w3schools.com/python/python_datatypes.asp'
'https://www.programiz.com/python-programming/variables-datatypes'