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

## Test cases
testGenerateRandomSequence()
testGenerateH()
testAddNoise()
testBlindOracle()
