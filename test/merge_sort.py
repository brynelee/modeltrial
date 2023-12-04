def merge_sort(arr):
    # Base case: if the array has only one element, it is already sorted.
    if len(arr) <= 1:
        return arr
    # Split the array into two halves: the first half is the left half of the array,
    # and the second half is the right half of the array.
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    # Recursively sort each half using merge sort.
    left = merge_sort(left)
    right = merge_sort(right)
    # Merge the two sorted halves into a single sorted array.
    return merge(left, right)


def merge(left, right):
    # Initialize an empty list to store the merged sorted array.
    merged = []
    # Initialize two pointers for each half of the array: one for the left half
    # and one for the right half.
    l = r = 0
    # Loop until one of the halves has exhausted its elements.
    while l < len(left) and r < len(right):
        # If the left half is larger, add its first element to the merged array
        if left[l] > right[r]:
            merged.append(left[l])
            l += 1
        else:
            merged.append(right[r])
            r += 1
    # Add any remaining elements from the left half to the merged array.
    while l < len(left):
        merged.append(left[l])
        l += 1
    # Add any remaining elements from the right half to the merged array.
    while r < len(right):
        merged.append(right[r])
        r += 1
    return merged


# Test the merge sort algorithm on a list of numbers.
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))