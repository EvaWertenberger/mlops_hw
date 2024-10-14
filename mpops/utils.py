def div_numbers(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("b cannot be zero")
    return a // b

def summ_numbers(a: int, b:int) -> int:
    return a + b