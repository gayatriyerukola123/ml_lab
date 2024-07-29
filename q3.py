def matrix_multiply(A, B):
    n = len(A)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return result
def matrix_power(A, m):
    if m < 1:
        raise ValueError("The exponent m must be a positive integer.")
    n = len(A)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = A
    while m > 0:
        if m % 2 == 1:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        m //= 2
    
    return result

def get_matrix_input():
    try:
        n = int(input("Enter the size of the square matrix (n): "))
        matrix = []
        print("Enter the matrix elements row by row, separated by spaces:")
        for i in range(n):
            row = list(map(float, input().split()))
            if len(row) != n:
                raise ValueError(f"Each row must have exactly {n} elements.")
            matrix.append(row)
        
        return matrix
    except ValueError as e:
        print(f"Error: {e}")
        return None

def get_integer_input(prompt):
    
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                raise ValueError("The integer must be positive.")
            return value
        except ValueError as e:
            print(f"Error: {e}. Please enter a positive integer.")

print("Matrix Exponentiation Program")

matrix = get_matrix_input()
if matrix is not None:
    exponent = get_integer_input("Enter the positive integer m: ")
    
    try:
        result = matrix_power(matrix, exponent)
        print("Matrix A^m is:")
        for row in result:
            print(" ".join(map(str, row)))
    except ValueError as e:
        print(f"Error: {e}")
