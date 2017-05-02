Tests
=====

LibSPN uses the [unittest](https://docs.python.org/3.5/library/unittest.html)
unit testing framework and defines multiple tests in the `libspn/tests` folder.

In order to run all tests, we recommend using
[nose2](https://github.com/nose-devs/nose2) in the main project directory, e.g.
`nose2 -v`. You can also run `make test` instead.

To run a specific test case, just execute the corresponding `test_<case>.py`
file in the `libspn/tests` folder.
