import sys, math

def main():
    T_and_Ns = sys.stdin.read().split()
    T = int(T_and_Ns[0])
    Ns = list(map(int, T_and_Ns[1:T+1]))
    if not Ns:
        return
    max_N = max(Ns)
    sieve = [True] * (max_N + 1)
    sieve[0] = sieve[1] = False
    sqrt_max = int(math.isqrt(max_N)) + 1
    for i in range(2, sqrt_max):
        if sieve[i]:
            sieve[i*i:max_N+1:i] = [False] * len(sieve[i*i:max_N+1:i])
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    P_valid = [False] * (max_N + 1)
    for i in range(len(primes)):
        R = primes[i]
        for j in range(i, len(primes)):
            Q = primes[j]
            P = Q - R
            if P < 2:
                continue
            if P > max_N:
                break
            if sieve[P]:
                P_valid[P] = True
    prefix_counts = [0] * (max_N + 1)
    count = 0
    for i in range(2, max_N + 1):
        if P_valid[i]:
            count += 1
        prefix_counts[i] = count
    output = []
    for idx, N in enumerate(Ns, 1):
        res = prefix_counts[N] if N >= 2 else 0
        output.append(f"Case #{idx}: {res}")
    print('\n'.join(output))

if __name__ == "__main__":
    main()