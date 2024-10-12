def mod_inverse(a, p):
    return pow(a, p - 2, p)

def expected_days_optimized(W, G, L, mod):
    if G >= W:
        return 0
    if L == 0 or L >= W - G:
        return W - G
    
    X = W - G
    half_mod = mod_inverse(2, mod)
    
    geometric_sum = (pow(2, X, mod) - 1 + mod) % mod
    extra_multiplier = mod_inverse((pow(2, X, mod) - 1), mod)
    
    result = geometric_sum * half_mod % mod
    result = result * extra_multiplier % mod
    
    return result

if __name__ == "__main__":
    mod = 998244353
    T = int(input())
    for case_number in range(1, T + 1):
        W, G, L = map(int, input().split())
        result = expected_days_optimized(W, G, L, mod)
        print(f"Case #{case_number}: {result}")