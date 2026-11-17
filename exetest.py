
###
## A simple Python program to test Python execution
###

import sys

# join all command-line arguments after the script name
if len(sys.argv) > 1:
    name = ' '.join(sys.argv[1:])
else:
    name = 'you'

message = f'Hello, {name}!'

# padding and dimensions
padding = 4
width = len(message) + 2 * padding
border = '*' * width
empty_line = '*' + ' ' * (width - 2) + '*'

# print the rectangle
print(border)
print(empty_line)
print('*' + ' ' * ((width - 2 - len(message)) // 2) + message + ' ' * ((width - 2 - len(message) + 1) // 2) + '*')
print(empty_line)
print(border)

from pathlib import Path
p = Path('.')
print(p.resolve())