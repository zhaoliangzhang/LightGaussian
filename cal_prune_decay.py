x = 0.1
y = 0.95

left = 1
for i in range(10):
    left = left * ((1-x)**y)
    print(left)