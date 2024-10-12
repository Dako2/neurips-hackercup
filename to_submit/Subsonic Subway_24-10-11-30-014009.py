def solve_sonic_delivery(test_cases):
    results = []
    for case_index, (N, stations) in enumerate(test_cases):
        global_max_min_speed = 0
        global_min_max_speed = float('inf')
        is_possible = True
        for i in range(N):
            A_i, B_i = stations[i]
            if B_i == 0 or A_i == 0:
                is_possible = False
                break
            min_speed_i = (i + 1) / B_i
            max_speed_i = (i + 1) / A_i
            global_max_min_speed = max(global_max_min_speed, min_speed_i)
            global_min_max_speed = min(global_min_max_speed, max_speed_i)
        if not is_possible or global_max_min_speed > global_min_max_speed:
            results.append(f"Case #{case_index + 1}: -1")
        else:
            results.append(f"Case #{case_index + 1}: {global_max_min_speed:.6f}")
    return results

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    T = int(data[0])
    index = 1
    test_cases = []
    for _ in range(T):
        N = int(data[index])
        stations = []
        index += 1
        for i in range(N):
            A_i = int(data[index])
            B_i = int(data[index + 1])
            stations.append((A_i, B_i))
            index += 2
        test_cases.append((N, stations))
    results = solve_sonic_delivery(test_cases)
    for result in results:
        print(result)