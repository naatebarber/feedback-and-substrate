eps = 0.3
c = 0
while True:
    eps *= 0.99999
    if c % 10000 == 0:
        print(eps, c)
    c += 1
