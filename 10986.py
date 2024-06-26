import sys
from itertools import accumulate
input = sys.stdin.readline

n, m = map(int, input().split())
a = list(map(int, input().split()))

# 접두사 합을 계산한 후 m으로 모듈로 연산
prefix_sums = list(accumulate(a))
mod_count = [0] * m

# 접두사 합의 모듈로 결과를 계산하고 카운트
for i in range(n):
    mod = prefix_sums[i] % m
    prefix_sums[i] = mod  # 계산된 모듈로를 다시 접두사 합 배열에 저장
    mod_count[mod] += 1

# 부분 배열의 수 계산
# 인덱스 0에서 시작하는 부분 배열을 위한 초기 카운트
cnt = mod_count[0]

# 같은 모듈로 결과를 가진 인덱스 조합을 더함
for count in mod_count:
    if count > 1:
        cnt += count * (count - 1) // 2

print(cnt)
