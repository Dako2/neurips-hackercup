import sys
MOD = 998244353
def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    index = 1
    for test_case in range(1, T + 1):
        W = int(data[index])
        G = int(data[index + 1])
        L = int(data[index + 2])
        index += 3
        D = W - G
        D_mod = D % MOD
        term = (2 * (L % MOD) + 1) % MOD
        E = (D_mod * term) % MOD
        print(f"Case #{test_case}: {E}")
if __name__ == "__main__":
    main()