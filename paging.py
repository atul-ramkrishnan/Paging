import random
import math
import os

# ---------------------------------------------------
# ALGORITHMS
# ---------------------------------------------------

def _initializeLRUCache(k):
    return [None] * k

def _initializeBlindOracleCache(k):
    return [(None, float('inf')) for _ in range(k)]

def generateRandomSequence(k, N, n, epsilon):
    """
    Generates a random sequence of pages.

    Parameters:
        k (int): The size of the cache
        N (int): The range of the page request [1...N]
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

def blindOracle(k, seq, hseq, initCache=None):
    """
    Runs the blindOracle algorithm.

    Parameters:
        k (int): The size of the cache
        seq (list): A list of integers representing the page requests
        hseq (list): A list of integers representing the predicted h values

    Returns:
        int: The number of page faults
    """

    if initCache is None:
        cache = _initializeBlindOracleCache(k)
    else:
        cache = initCache
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

def LRU(k, seq, initCache=None):
    """
    Runs the LRU algorithm.

    Parameters:
        k (int): The size of the cache
        seq (list): A list of integers representing the page requests

    Returns:
        int: The number of page faults
    """

    if initCache is None:
        cache = _initializeLRUCache(k)
    else:
        cache = initCache

    numPageFaults = 0
    for page in seq:
        if page in cache:
            # Cache hit, move page to the end to mark it as recently used
            cache.remove(page)
            cache.append(page)
        else:
            # Cache miss
            numPageFaults += 1
            if len(cache) >= k:
                # Cache is full, remove the least recently used item
                cache.pop(0)
            cache.append(page)

    return numPageFaults

def combinedAlg(k, seq, hseq, thr):
    """
    Runs the "Combined" algorithm.

    Parameters:
        k (int): The size of the cache
        seq (list): A list of integers representing the page requests
        hseq (list): A list of integers representing the predicted h values
        thr (float): Threshold parameter between 0 and 1

    Returns:
        int: The number of page faults
    """

    numPageFaults = 0
    f1, f2 = 0, 0  # Page faults for LRU and blindOracle
    cacheLRU = _initializeLRUCache(k)
    cacheBlindOracle = _initializeBlindOracleCache(k)
    useLRU = True  # Start with LRU

    for i in range(len(seq)):
        page = seq[i:i+1]
        h_value = hseq[i:i+1]

        faultLRU = LRU(k, page, cacheLRU)
        f1 += faultLRU

        faultBlindOracle = blindOracle(k, page, h_value, cacheBlindOracle)
        f2 += faultBlindOracle

        # Increment numPageFaults by the number of faults incurred by the active strategy
        numPageFaults += faultLRU if useLRU else faultBlindOracle

        # Check if a switch is needed
        if useLRU and f1 > (1 + thr) * f2:
            useLRU = False
            numPageFaults += k  # Account for cache switch
        elif not useLRU and f2 > (1 + thr) * f1:
            useLRU = True
            numPageFaults += k  # Account for cache switch
        
    return numPageFaults

# ---------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------

def runSingleTrial(k, N, n, epsilon, gamma, omega):
    """
    Runs a single trial of paging algorithms using generated page request sequences
    and associated h-values, both original and with added noise, to evaluate
    performance in terms of page faults.

    Parameters:
        k (int): The size of the cache for each paging algorithm.
        N (int): The maximum page number in the page request sequence (range 1 to N).
        n (int): The total number of page requests to generate for the sequence.
        epsilon (float): The probability factor controlling the locality in the page sequence.
        gamma (float): The probability with which noise is added to each h-value in the sequence.
        omega (int): The maximum amount of noise that can be added to each h-value.

    Returns:
        dict: A dictionary containing the number of page faults incurred by each algorithm.
    """
    seq = generateRandomSequence(k, N, n, epsilon)
    hseq = generateH(seq)
    hseqNoisy = addNoise(hseq, gamma, omega)
    thr = 0.5

    results = {
        'opt': blindOracle(k, seq, hseq),
        'blindoracle': blindOracle(k, seq, hseqNoisy),
        'lru': LRU(k, seq),
        'combined': combinedAlg(k, seq, hseqNoisy, thr)
    }
    return results

def runBatchOfTrials(numTrials, k, N, n, epsilon, gamma, omega):
    """
    Runs a batch of trials with specified paging and noise parameters, 
    and aggregates the results across all trials to compute average page faults for each paging algorithm.

    Parameters:
        numTrials (int): Number of trials to run.
        k (int): The size of the cache used in each paging algorithm.
        N (int): The upper limit of the page request range [1...N].
        n (int): The total number of page requests to be generated in each trial.
        epsilon (float): Parameter controlling the amount of locality in the page request sequence.
        gamma (float): Noise parameter representing the probability with which noise is added to the paging sequence.
        omega (int): Noise parameter specifying the amplitude of noise to be added.

    Returns:
        dict: A dictionary containing the average number of page faults incurred by each algorithm across all trials,
              along with the experiment parameters used for the batch.
    """
    results = []
    for _ in range(numTrials):
        trial_result = runSingleTrial(k, N, n, epsilon, gamma, omega)
        results.append(trial_result)
    
    average_results = {
        'numTrials': numTrials,
        'k': k,
        'N': N,
        'n': n,
        'epsilon': epsilon,
        'gamma': gamma,
        'omega': omega
    }
    
    average_results.update({algorithm: 0 for algorithm in results[0]})

    for result in results:
        for algorithm in result:
            average_results[algorithm] += result[algorithm] / numTrials
    
    return average_results


def saveResultsToCSV(results, filename="data/results.csv"):
    """
    Saves the paging algorithm results and their corresponding experimental parameters to a CSV file.

    Parameters:
        results (list of dict): List of dictionaries containing the aggregate results for each algorithm along with the experiment parameters.
        filename (str): The name of the file to save the results to.

    Returns:
        None
    """
    if not results:
        return

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    headers = results[0].keys()
    with open(filename, 'w') as f:
        f.write(','.join(headers) + '\n')
        for result in results:
            row = [str(result[header]) for header in headers]
            f.write(','.join(row) + '\n')

# ---------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------

def testGenerateRandomSequence():
    """
    Test function for generateRandomSequence function.

    Returns:
        None
    """
    print("========== Testing generateRandomSequence ==========")
    trials = 10     # Number of times to run the test for statistical significance
    for _ in range(trials):
        k, N, n, epsilon = 5, 100, 10, 0.5
        sequence = generateRandomSequence(k, N, n, epsilon)
        assert len(sequence) == n, "Incorrect length of sequence generated"
        assert sequence[:k] == list(range(1, k + 1)), "Initial elements are incorrect"
        assert all(1 <= x <= N for x in sequence), "Elements in sequence out of range"

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

    trials = 10    # Number of times to run the test for statistical significance
    hseq = [12, 4, 5, 7, 8, 9, 10, 12, 12, 11, 12]
    num_steps = 10
    for _ in range(trials):
        for gamma in [i / num_steps for i in range(num_steps + 1)]:
            for omega in range(1, 10):
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

    k = 1
    seq = [1, 2, 3, 2, 4, 3]
    hseq = [7, 4, 9, 7, 7, 7]
    pageFaults = blindOracle(k, seq, hseq)
    assert pageFaults == 6, "Incorrect number of page faults"

    k = 6
    seq = [1, 2, 3, 2, 4, 3]
    hseq = [7, 4, 9, 7, 7, 7]
    pageFaults = blindOracle(k, seq, hseq)
    assert pageFaults == 4, "Incorrect number of page faults"

    k = 3
    seq = list(range(1, 10))
    hseq = [10] * 9
    pageFaults = blindOracle(k, seq, hseq)
    assert pageFaults == 9, "Incorrect number of page faults"

    print("All tests passed.")

def testLRU():
    """
    Test function for LRU function.

    Returns:
        None
    """
    print("========== Testing LRU ==========")

    k = 4
    seq = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]
    pageFaults = LRU(k, seq)
    assert pageFaults == 6, "Incorrect number of page faults"

    k = 1
    seq = list(range(1, 11))
    pageFaults = LRU(k, seq)
    assert pageFaults == 10, "Incorrect number of page faults"

    k = 10
    seq = [1] * 100
    pageFaults = LRU(k, seq)
    assert pageFaults == 1, "Incorrect number of page faults"

    print("All tests passed.")

def testcombinedAlg():
    """
    Test function for combinedAlg function.

    Returns:
        None
    """
    print("========== Testing combinedAlg ==========")
    k = 3
    seq = [1, 2, 3, 2, 1, 4, 3, 5, 6, 4, 3, 5, 3, 5, 6, 7, 2, 1, 5, 7]
    hseq = generateH(seq)
    thr = 0
    pageFaultsCombined = combinedAlg(k, seq, hseq, thr)
    assert pageFaultsCombined == 14, "Incorrect number of page faults"

    thr = 0.3
    pageFaultsCombined = combinedAlg(k, seq, hseq, thr)
    assert pageFaultsCombined == 15, "Incorrect number of page faults"

    thr = 1
    pageFaultsCombined = combinedAlg(k, seq, hseq, thr)
    assert pageFaultsCombined == 16, "Incorrect number of page faults"

    hseqNeg = [-h for h in hseq]
    thr = 0
    pageFaultsCombined = combinedAlg(k, seq, hseqNeg, thr)
    assert pageFaultsCombined == 16, "Incorrect number of page faults"

    print("All tests passed.")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
    
def testCustom():
    average_results = [
        {
            'numTrials': 10,
            'k': 5,
            'N': 10,
            'n': 100,
            'epsilon': 0.5,
            'gamma': 0.6,
            'omega': 0.7,
            'opt' : 30,
            'blindOracle': 40

        },
        {
            'numTrials': 20,
            'k': 50,
            'N': 10,
            'n': 100,
            'epsilon': 0.5,
            'gamma': 0.6,
            'omega': 0.7,
            'opt' : 30,
            'blindOracle': 40

        },
    ]
    saveResultsToCSV(average_results)

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
    testLRU()
    testcombinedAlg()
    testCustom()



if __name__ == "__main__":
    main()
