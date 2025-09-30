nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat_list = [item for item in sublist for sublist in nested_list]

print(flat_list)
# Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]