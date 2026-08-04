Unit testing coverage report.
--------------------------------------------------------------------------------------------------------
.coverage file included in documents folder

====================================================================================================== test session starts ======================================================================================================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
plugins: anyio-4.12.1, Faker-40.15.0
collected 20 items                                                                                                                                                                                                               

tests\algorithms\test_neural_net.py ......                                                                                                                                                                                 [ 30%]
tests\algorithms\test_replaybuffer.py ...                                                                                                                                                                                  [ 45%]
tests\game\test_dice.py ....                                                                                                                                                                                               [ 65%]
tests\game\test_scoreboard.py ...                                                                                                                                                                                          [ 80%]
tests\game\test_yatzy.py ....                                                                                                                                                                                              [100%]

====================================================================================================== 20 passed in 0.57s =======================================================================================================

--------------------------------------------------------------------------------------------------------


What was tested and how?

The core game functionalities of yatzy.py, dice.py and scoreboard.py were tested using unit tests with sample inputs and outputs
The core algorithm, being the deep-q-network was tested with sanity checks including that forward pass gives valid output values, weights change after update step,
Q-score increases after multiple updates using the same move and reward and that at the end of the game the bellman equation only gives the reward without taking
into account future actions and Q-values. For these tests to pass the neural network implementation needs to be working in a solid way, especially for the Q-value increasing test
proves that many functions are working correctly together and learning is happening.

--------------------------------------------------------------------------------------------------------


What types of inputs were used for testing?

-Inputs were game states that the game can produce and agent could encounter during normal gameplay.

--------------------------------------------------------------------------------------------------------

How can the tests be reproduced?

python -m coverage run -m pytest
