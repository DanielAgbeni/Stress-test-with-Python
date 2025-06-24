import random
import time

TOTAL_SIZE = 50_000_000

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def main():
    print('Generating 50 million random integers...')
    arr = [random.randint(0, TOTAL_SIZE) for _ in range(TOTAL_SIZE)]
    print('Sorting...')
    start = time.time()
    sorted_arr = quicksort(arr)
    end = time.time()
    print(f'Sorted in {end - start:.2f} seconds')

if __name__ == '__main__':
    main()
