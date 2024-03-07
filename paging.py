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
        list: A list of integers representing the generated sequence of pages
    """
    p = [None] * n
    p[0 : k] = range(1, k + 1)
    L = set(range(1, k + 1))

    for i in range(k, n):
        x = random.choice(list(L))
        y = random.choice(list(set(range(1, N + 1)).difference(L)))

        if random.random() < epsilon:
            # With probability epsilon, set p[i] to x
            p[i] = x
        else:
            # With probability 1 - epsilon, set p[i] to y
            p[i] = y
            # Update the set L
            L.remove(x)
            L.add(y)
    
    return p

def generateH(seq):
    """
    Generates h-values for a sequence of page requests.

    Parameters:
        seq (list): A list of integers representing a sequence of page requests
    
    Returns:
        list: A list of integers representing h-values
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
    Adds noise to the h-values.

    Parameters:
        hseq (list): A list of integers representing the h-values
        gamma (float): Noise parameter representing the probability with which noise is added
        omega (int): Noise parameter representing the amount of noise to be added.
    
    Returns:
        list: A list of integers representing h-values with added noise
    """
    hHat = hseq.copy()
    for i in range(len(hseq)):
        if random.random() >= 1 - gamma:
            # With probability 1 - gamma, add noise to the h-values
            l = max(i + 2, hseq[i] - math.floor(omega / 2))
            hHat[i] = random.randint(l, l + omega)

    return hHat

def blindOracle(k, seq, hseq):
    """
    Runs the blindOracle algorithm.

    Parameters:
        k (int): The size of the cache
        seq (list): A list of integers representing the page requests
        hseq (list): A list of integers representing the predicted h values

    Returns:
        int: The number of page faults
    """
    cache = [(None, float('inf')) for _ in range(k)]
    numPageHits = 0
    
    for p, h in zip(seq, hseq):
        pageHit = False
        hMax = 0
        idxMax = -1

        # Iterate over cache to find if element is already in cache or to find the element with maximum h-value
        for idx, (elem, elemH) in enumerate(cache):
            if p == elem:  # Page hit
                cache[idx] = (p, h)  # Update h-value in cache
                pageHit = True
                numPageHits += 1
                break
            if elemH > hMax:  # Find the max h-value in cache and its index
                hMax = elemH
                idxMax = idx

        if not pageHit:
            cache[idxMax] = (p, h)

    # Number of Page Faults
    return len(seq) - numPageHits

def testGenerateRandomSequence():
    """
    Test function for generateRandomSequence function.

    Returns:
        None
    """
    print("========== Testing generateRandomSequence ==========")
    k, N, n, epsilon = 5, 100, 10, 0.5
    sequence = generateRandomSequence(k, N, n, epsilon)
    assert len(sequence) == n, "Incorrect length of sequence generated"
    assert sequence[:k] == list(range(1, k + 1)), "Initial elements are incorrect"
    assert all(1 <= x <= N for x in sequence), "Elements in sequence out of range"

    trials = 10     # Number of times to run the test for statistical significance
    for _ in range(trials):
        k, N, n, epsilon = 5, 1000, 100, 1.0
        sequence = generateRandomSequence(k, N, n, epsilon)
        assert all(1 <= x <= k for x in sequence), "Invalid element in sequence"

    print("All tests passed.")

def testGenerateH():
    """
    Test function for generateH function.

    Returns:
        None
    """
    print("========== Testing generateH ==========")
    sequence = [1, 2, 3, 2, 4, 5, 1, 2]
    h = generateH(sequence)
    assert h == [7, 4, 9, 8, 9, 9, 9, 9], "Incorrect h-values"

    sequence = [1, 2, 3, 2, 3, 4, 2, 3, 4, 2, 2]
    h = generateH(sequence)
    assert h == [12, 4, 5, 7, 8, 9, 10, 12, 12, 11, 12]

    sequence = [1, 1, 1, 1, 1]
    h = generateH(sequence)
    assert h == [2, 3, 4, 5, 6], "Incorrect h-values"

    sequence = [1, 2, 3, 4, 5]
    h = generateH(sequence)
    assert h == [6, 6, 6, 6, 6], "Incorrect h-values"

    sequence = [5, 4, 3, 2, 1]
    h = generateH(sequence)
    assert h == [6, 6, 6, 6, 6], "Incorrect h-values"
    
    print("All tests passed.")

def testAddNoise():
    """
    Test function for addNoise function.

    Returns:
        None
    """
    print("========== Testing addNoise ==========")
    hseq = [12, 4, 5, 7, 8, 9, 10, 12, 12, 11, 12]
    gamma, omega = 0, 10
    hseqNoisy = addNoise(hseq, gamma, omega)
    assert hseq == hseqNoisy, "Noise added to h-values when gamma = 0"

    hseq = [12, 4, 5, 7, 8, 9, 10, 12, 12, 11, 12]
    gamma, omega = 1, 10
    trials = 10    # Number of times to run the test for statistical significance
    for _ in range(trials):
        hHat = addNoise(hseq, gamma, omega)
        for i, (original, noisy) in enumerate(zip(hseq, hHat)):
            l = max(i + 2, original - math.floor(omega / 2))
            assert(l <= noisy <= l + omega), "Noise added not within expected range"

    print("All tests passed.")

def testBlindOracle():
    """
    Test function for blindOracle function.

    Returns:
        None
    """
    print("========== Testing blindOracle ==========")
    k = 3
    seq = [1, 2, 3, 2, 4, 3]
    hseq = [7, 4, 6, 7, 7, 7]
    k = 3
    pageFaults = blindOracle(k, seq, hseq)
    assert pageFaults == 4, "Incorrect number of page faults"

    k = 3
    seq = [1, 2, 3, 2, 4, 3]
    hseq = [7, 4, 9, 7, 7, 7]
    pageFaults = blindOracle(k, seq, hseq)
    assert pageFaults == 5, "Incorrect number of page faults"

    k = 3
    seq = list(range(1, 10))
    hseq = [10] * 9
    pageFaults = blindOracle(k, seq, hseq)
    assert pageFaults == 9, "Incorrect number of page faults"

    print("All tests passed.")

def main():
    """
    Main function. Runs all the tests.

    Returns:
        None
    """
    testGenerateRandomSequence()
    testGenerateH()
    testAddNoise()
    testBlindOracle()


if __name__ == "__main__":
    main()
