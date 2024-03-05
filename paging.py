import random
import math


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

def generateH(seq):
    n = len(seq)
    h = [n + 1] * n
    last_positions = {}  # Dictionary to store the last position of each element

    # Traverse the sequence in reverse
    for i in range(n - 1, -1, -1):
        element = seq[i]
        if element in last_positions:
            # If the element has been seen before, update h[i] with the position of the next occurrence
            h[i] = last_positions[element] + 1
        # Update the last position of the current element
        last_positions[element] = i

    return h


def addNoise(hseq, gamma, omega):
    h_hat = hseq.copy()
    for i in range(len(hseq)):
        if random.random() >= 1 - gamma:
            l = max(i + 2, hseq[i] - math.floor(omega / 2))
            h_hat[i] = random.randint(l, l + omega)

    return h_hat

def blindOracle(k, seq, hseq):
    pass

def testGenerateH():
    seq = [1, 2, 3, 2, 4, 5, 1, 2]
    h = generateH(seq)
    assert h == [7, 4, 9, 8, 9, 9, 9, 9]

    seq = [1, 1, 1, 1, 1]
    h = generateH(seq)
    assert h == [2, 3, 4, 5, 6]

    seq = [1, 2, 3, 4, 5]
    h = generateH(seq)
    assert h == [6, 6, 6, 6, 6]

    seq = [5, 4, 3, 2, 1]
    h = generateH(seq)
    assert h == [6, 6, 6, 6, 6]

def testAddNoise():
    hseq = [7, 4, 9, 8, 9, 9, 9, 9]
    hseqNoisy = addNoise(hseq, 0.7, 1)
    print(hseqNoisy)

def main():
    testGenerateH()
    testAddNoise()


if __name__ == "__main__":
    main()
