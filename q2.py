def count_pairs_with_sum(lst, target_sum):
    lst.sort()  
    left = 0
    right = len(lst) - 1
    pairs_count = 0
    while left < right:
        current_sum = lst[left] + lst[right]
        if current_sum == target_sum:
            pairs_count += 1
            left += 1
            right -= 1
        elif current_sum < target_sum:
            left += 1
        else:
            right -= 1
    return pairs_count
lst = [2, 7, 4, 1, 3, 6]
target_sum = 10
print("Number of pairs with sum equal to", target_sum, "is:", count_pairs_with_sum(lst, target_sum))
