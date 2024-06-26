import sys
input = sys.stdin.readline

n, m = map(int, input().split())
a = [list(map(int, input().split())) for _ in range(n)]
b = [list(map(int, input().split())) for _ in range(m)]

# 접두사 합 배열 초기화
sum_a = [[0] * (n + 1) for _ in range(n + 1)]

# 접두사 합 계산
for i in range(1, n + 1):
    for j in range(1, n + 1):
        sum_a[i][j] = a[i-1][j-1] + sum_a[i-1][j] + sum_a[i][j-1] - sum_a[i-1][j-1]

# 쿼리에 따른 부분 행렬 합 계산 및 출력
for x1, y1, x2, y2 in b:
    result = sum_a[x2][y2] - sum_a[x1-1][y2] - sum_a[x2][y1-1] + sum_a[x1-1][y1-1]
    print(result)