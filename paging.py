import random


def generateRandomSequence(k, N, n, epsilon):
    p = [None] * n
    p[0 : k] = range(1, k + 1)
    L = set(range(1, k + 1))

    for i in range(k, n):
        x = random.choice(list(L))
        y = random.choice(list(set(range(1, N + 1)).difference(L)))

        if random.random() < epsilon:
            p[i] = x
        else:
            p[i] = y
            L.remove(x)
            L.add(y)
    
    return p

def generate(seq):
    pass

def addNoise(hseq, gamma, omega):
    pass

def blindOracle(k, seq, hseq):
    pass

def main():
    pass


if __name__ == "__main__":
    main()
