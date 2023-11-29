import sys
from tqdm import tqdm

input_filename = sys.argv[1]
output_filename = 'reprocessed-' + input_filename

print(input_filename, output_filename)

with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
    prev_second_val = None
    prev_line = None
    for line in tqdm(input_file):
        first_val, second_val, rest = line.split(',', 2)
        if first_val != prev_second_val:
            if prev_line is not None:
                remaining_prev_vals = prev_line.strip().split(',')[1:]
                print(f'FINISHING {len(remaining_prev_vals)}')
                for remaining_prev_val in remaining_prev_vals:
                    output_file.write(remaining_prev_val + '\n')
            print('START')
            output_file.write('START' + '\n')
        prev_second_val = second_val
        prev_line = line
        output_file.write(first_val + '\n')

print('Done.')
