# Paging

## Implementation details
The project is implemented in Python 3.12.2
The project runs the experiments and saves the results as a .csv file.
Then you can plot the results of the experiments as a graph. Plotting the graphs requires the following packages --
  matplotlib 3.8.4 
  pandas 2.1.1  

  The experiments parameters are --
  n = 10000
  regime1 = {
        'k': 10,
        'N': 100,
        'n': 10000,
        'epsilon': 0.5,
        'omega': 500,
        'gamma': 0.3
    }

    regime2 = {
        'k': 10,
        'N': 100,
        'n': 10000,
        'epsilon': 0.5,
        'omega': 1000,
        'gamma': 0.99
    }

## Usage
To run the code, first you need to go the project directory.

To run the script, use one of the following commands depending on the function you wish to execute
1. Run tests
  python paging.py runTests

2. Find trends
  python paging.py findTrends

3. Plot trends
  python paging.py plotTrends


## Expected output
1. python paging.py runTests

  In the case of no errors, the output should be:
  ========== Testing generateRandomSequence ==========
  All tests passed.
  ========== Testing generateH ==========
  All tests passed.
  ========== Testing addNoise ==========
  All tests passed.
  ========== Testing blindOracle ==========
  All tests passed.
  ========== Testing LRU ==========
  All tests passed.
  ========== Testing combinedAlg ==========
  All tests passed.

  If there are any errors, the assertion will fail with an appropriate message. For instance,
  Traceback (most recent call last):
    File "/Users/atulramkrishnan/Documents/GitHub/Paging/paging.py", line 241, in <module>
      main()
    File "/Users/atulramkrishnan/Documents/GitHub/Paging/paging.py", line 237, in main
      testBlindOracle()
    File "/Users/atulramkrishnan/Documents/GitHub/Paging/paging.py", line 211, in testBlindOracle
      assert pageFaults == 4, "Incorrect number of page faults"
  AssertionError: Incorrect number of page faults

2. python paging.py findTrends
  This function runs silently without console outputs and performs a series of experimental runs to identify how different parameters influence the performance of paging algorithms.
  The results from these experiments are saved into CSV files, and no direct output is printed to the console. Here's what to expect:
  Files Created

  The function will generate several CSV files, each corresponding to a different trend analysis. These files will be saved in the 'data/' directory:

    trend1.csv: Contains the results of experiments varying the cache size ('k'). The file records outcomes for each set value of 'k' from 10 to 100 in increments of 10.

    trend2.csv: Contains the results of experiments varying the noise amplitude ('omega'). The file records outcomes for each set value of 'omega' from 0 to 2000 in increments of 200.

    trend3.csv: Contains the results of experiments varying the locality parameter ('epsilon'). The file records outcomes for each set value of 'epsilon' from 0 to 1 in increments of 0.1.

    trend4.csv: Contains the results of experiments varying the noise probability ('gamma'). The file records outcomes for each set value of 'gamma' from 0 to 1 in increments of 0.1.

3. python paging.py plotTrends
  This function generates plots based on the data from various trend analyses and saves these plots as image files.
  Note that this function expects to find the necessary .csv files in the 'data/' folder. Please ensure that all required .csv files are placed in the correct subdirectory before running the function.
  It operates silently without console outputs unless there's an error. Here's what to expect:
  Files Created

  The function will generate several image files, each corresponding to a different trend analysis. These files are saved in the 'images/' directory:

    trend1.png: A plot showing the results of varying cache size ('k'). It visualizes how the number of page faults changes with different cache sizes for each paging algorithm.

    trend2.png: A plot displaying the effects of varying noise amplitude ('omega'). This plot illustrates the impact of increasing noise on the number of page faults.

    trend3.png: A plot detailing the influence of the locality parameter ('epsilon'). It shows how different levels of locality affect page fault rates.

    trend4.png: A plot focused on the changes in page faults as the noise probability ('gamma') is adjusted.


## Unit tests
testGenerateRandomSequence()
The test cases in testGenerateRandomSequence focus on validating the properties of the generated random sequence.
For instance, the test check if the numbers generated as part of the sequence are in the range [1, N].
The test also checks if the numbers generated in the edge case where eps = 1.0 are in the range [1, k].
To account for the inherent randomness in the generateRandomSequence function, the tests are run 'trial' times.

testGenerateH()
The test cases in testGenerateH validate generated h-values against expected h-values for a set of sequences.

testAddNoise()
The test cases in testAddNoise test that the added noise is the expected range for a set of values of (gamma, omega) on a particular sequence of hseq
As in testGenerateRandomSequence, to account for the inherent randomness in the generateRandomSequence function, the tests are run 'trial' times.

testBlindOracle()
The test cases in testBlindOracle test the number of page faults against the expected number of page faults for a set of (k, seq, hseq) tuples.

testLRU()
The test cases in testLRU test the number of page faults against the expected number of page faults for a set of (k, seq) tuples.

testcombinedAlg()
The test cases in testBlindOracle test the number of page faults against the expected number of page faults for a set of (k, seq, hseq, thr) tuples.
Most test cases run the combinedAlg for the the h-values without any added noise, essentially making the blindOracle into OPT. In this case, it would
benefit the combinedAlg to switch from LRU to blindOracle as soon as possible. In other words, a lower threshold should yield a lower number of page faults.
There is also a test case that runs combinedAlg on a "bad" list of h-values, generated by negating the h-values. In this case, it is expected that the combinedAlg
would yield a higher number of page faults for the same threshold.