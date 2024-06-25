from itertools import accumulate
n, m=list(map(int, input().split()))
arr=list(map(int,input().split()))
a=[]
for i in range(m):
    a.append(list(map(int,input().split())))
sum_arr=list(accumulate(arr,initial=0))
for i in range(m):
    print(sum_arr[a[i][1]] - sum_arr[a[i][0]-1])