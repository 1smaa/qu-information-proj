from mpmath import factorial
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_factorial(n: int) -> int:
    return factorial(n,exact=False) 

@lru_cache(maxsize=None)
def cached_hermite(n: int,z: complex):
    if(n==0): return 1
    elif(n==1): return 2*z
    else: return 2*z*cached_hermite(n-1,z)-2*(n-1)*cached_hermite(n-2,z)