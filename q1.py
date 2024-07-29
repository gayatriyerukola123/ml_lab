def count_pairs_with_sum(lst, target_sum):
    seen = set()         
    pairs_count = 0     
    for number in lst:
        complement = target_sum - number
        if complement in seen:
            pairs_count += 1
        seen.add(number)
    return pairs_count
lst = [2, 7, 4, 1, 3, 6]
target_sum = 10
print("Number of pairs with sum equal to", target_sum, "is:", count_pairs_with_sum(lst, target_sum))
def count_pairs_with_sum(lst, target_sum):
    seen = set()         
    pairs_count = 0     
    for number in lst:
        complement = target_sum - number
        if complement in seen:
            pairs_count += 1
        seen.add(number)
    return pairs_count
lst = [2, 7, 4, 1, 3, 6]
target_sum = 10
print("Number of pairs with sum equal to", target_sum, "is:", count_pairs_with_sum(lst, target_sum))
