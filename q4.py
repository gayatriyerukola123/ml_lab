def count_highest_occurring_char(input_string):
    char_count = {}
    for char in input_string:
        if char.isalpha(): 
            char = char.lower() 
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
    if not char_count:
        return None, 0
    
    highest_char = max(char_count, key=char_count.get)
    highest_count = char_count[highest_char]
    
    return highest_char, highest_count

input_string = input("Enter a string: ")

highest_char, highest_count = count_highest_occurring_char(input_string)

if highest_char:
    print(f"The highest occurring character is '{highest_char}' with {highest_count} occurrences.")
else:
    print("No alphabetic characters found in the input.")
