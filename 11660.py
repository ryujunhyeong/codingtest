import sys
import itertools
import copy
input=sys.stdin.readline
n, m = map(int, input().split())
a=[]
b=[]
sum_map=[[0]*(n+1)]+[[0]*(n+1)]+[[0]*(n+1)]+[[0]*(n+1)]+[[0]*(n+1)]
a.append([0]*(n+1))
[a.append([0]+list(map(int,input().split()))) for _ in range(n)]
[b.append(list(map(int, input().split()))) for _ in range(m)]
print(a)
for i in range(1,n+1):
    for j in range(1,n+1):
        print(i, j)
        sum_map[i][j]=a[i-1][j]+a[i][j-1]
for i in range(m):
    print(sum_map[b[i][2]][b[i][3]]-sum_map[b[i][0]][b[i][1]])
    
print(sum_map)