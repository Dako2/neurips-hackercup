#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <utility>

using namespace std;

using ll = long long;
const int MAX_SIZE = 810;

int R, C;
ll K;
int grid[MAX_SIZE][MAX_SIZE];
vector<pair<int, int>> positions[800080];

// Update function for Fenwick Tree
void updateFenwick(int tree[], int index, int value) {
    while (index < MAX_SIZE) {
        tree[index] += value;
        index += index & -index;
    }
}

// Query function for range sum in Fenwick Tree
int queryFenwick(int tree[], int left, int right) {
    int sum = 0;
    for (int i = right; i > 0; i -= i & -i) sum += tree[i];
    for (int i = left - 1; i > 0; i -= i & -i) sum -= tree[i];
    return sum;
}

// Function to count valid pairs within distance L
ll countValidPairs(int L) {
    ll count = 0;
    int fenwickTree[MAX_SIZE] = {};

    // Iterate through each cell in the grid
    for (int row = 1; row <= R; ++row) {
        for (int col = 1; col <= C; ++col) {
            int topRow = max(1, row - L);
            int bottomRow = min(R, row + L);
            int leftCol = max(1, col - L);
            int rightCol = min(C, col + L);
            count += (bottomRow - topRow + 1) * (rightCol - leftCol + 1) - 1;
        }
    }

    // Process each unique owner (bunny) group
    for (int owner = 1; owner <= R * C; ++owner) {
        int j = 0;

        for (int i = 0; i < positions[owner].size(); ++i) {
            auto [row, col] = positions[owner][i];

            // Remove positions outside the current L distance range
            while (j < positions[owner].size() && 
                   (positions[owner][j].first < row - L || 
                   (positions[owner][j].first == row - L && positions[owner][j].second < col - L))) {
                updateFenwick(fenwickTree, positions[owner][j].second, -1);
                ++j;
            }

            int leftCol = max(1, col - L);
            int rightCol = min(C, col + L);
            count -= queryFenwick(fenwickTree, leftCol, rightCol) * 2;

            updateFenwick(fenwickTree, col, 1);
        }

        // Reset Fenwick Tree after processing each bunny type
        for (int i = 0; i < positions[owner].size(); ++i) {
            updateFenwick(fenwickTree, positions[owner][i].second, -1);
        }
    }
    return count;
}

// Binary search to find the smallest L that meets the required K
int findMinimumDistance() {
    int low = 1, high = max(R, C) - 1, result = -1;

    while (low <= high) {
        int mid = (low + high) / 2;
        if (countValidPairs(mid) >= K) {
            result = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return result;
}

void solve() {
    scanf("%d%d%lld", &R, &C, &K);
    for (int i = 1; i <= R; ++i)
        for (int j = 1; j <= C; ++j)
            scanf("%d", &grid[i][j]), positions[grid[i][j]].emplace_back(i, j);

    printf("%d\n", findMinimumDistance());

    // Clear positions after each test case
    for (int i = 1; i <= R * C; ++i) positions[i].clear();
}

int main() {
    int T;
    scanf("%d", &T);
    for (int testCase = 1; testCase <= T; ++testCase) {
        fprintf(stderr, "! %d\n", testCase);
        printf("Case #%d: ", testCase);
        solve();
    }
    return 0;
}
