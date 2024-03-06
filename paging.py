import random
import math


def generateRandomSequence(k, N, n, epsilon):
    """
    Generates a random sequence of pages.
    Parameters:
        k (int): The size of the cache
        N (int): The range of the page request [1..N]
        n (int): The number of page requests to be generated
        epsilon(float): Amount of locality in the sequence 
    
    Returns:
        list: A list of integers representing the generated sequence of pages.
    """

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
    """
    Generates "h" values for a sequence of page requests.

    Parameters:
        seq (list): A list of integers representing a sequence of page requests
    
    Returns:
        list: A list of integers representing "h" values
    """

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
    """
    Adds noise to the "h" sequence.

    Parameters:
        hseq (list): A list of integers representing the "h" sequence
        gamma (float): Noise parameter representing the probability with which noise is added
        omega (float): Noise parameter representing the amount of noise to be added.
    
    Returns:
        list: A list of integers representing "h" values with added noise
    """

    hHat = hseq.copy()
    for i in range(len(hseq)):
        if random.random() >= 1 - gamma:
            l = max(i + 2, hseq[i] - math.floor(omega / 2))
            hHat[i] = random.randint(l, l + omega)

    return hHat

def blindOracle(k, seq, hseq):
    """
    Parameters:
        k (int): The size of the cache
        seq (list): A list of integers representing the page requests
        hseq (list): A list of integers representing the predicted h values

    Returns:
        int: The number of page faults
    """

    cache = [(None, float('inf')) for _ in range(k)]
    numCacheHits = 0
    
    for p, h in zip(seq, hseq):
        cacheHit = False
        hMax = float('-inf')
        idxMax = -1

        # Iterate over cache to find if element is already in cache or to find the element with maximum h-value
        for idx, (elem, elemH) in enumerate(cache):
            if p == elem:  # Cache hit
                cache[idx] = (p, h)  # Update h-value in cache
                cacheHit = True
                numCacheHits += 1
                break
            if elemH > hMax:  # Find the max h-value in cache and its index
                hMax = elemH
                idxMax = idx

        if not cacheHit:
            cache[idxMax] = (p, h)

    return len(seq) - numCacheHits

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

def testBlindOracle():
    seq = [1, 2, 3, 2, 4, 3]
    hseq = [7, 4, 9, 7, 7, 7]
    pageFaults = blindOracle(3, seq, hseq)
    print(pageFaults)

def main():
    # testGenerateH()
    # testAddNoise()
    testBlindOracle()


if __name__ == "__main__":
    main()
