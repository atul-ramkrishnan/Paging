# Paging

## Implementation details
The project is implemented in Python 3.12.2


## How to run the main script
To run the code, first you need to go the project directory

Once you are in the paging directory, the code can be run using the command below:
python paging.py


## Expected output
The main function runs all the test cases. In the case of no errors, the output should be:
========== Testing generateRandomSequence ==========
All tests passed.
========== Testing generateH ==========
All tests passed.
========== Testing addNoise ==========
All tests passed.
========== Testing blindOracle ==========
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
The test cases in testBlindOracle test the number of page faults against the expected number of page faults for a set of (seq, hseq, k) tuples.
