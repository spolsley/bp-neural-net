bp-neural-net
===

About
---

This is a simple implementation of a back-propagating neural network.  Three variants of bp-neural-net have been written: Octave/Matlab, C++, and Python.  The Octave/Matlab version is a single script that is only capable of batch learning.  The C++ variant of bp-neural-net is the fastest.  It is capable of batch learning or live learning.  Likewise, the Python implementation supports batch and live learning.  However, Python is slightly slower than the other two versions.  Also, the Python version requires numpy be installed for its matrix operations.

Sample Data
---

Each version includes a sample set of data to run, the classifying of digit between 0 and 9.  The training ouptuts of each are designed to be nearly identical for easy comparison.  The C++ and Python versions include runners that train the network on the test data.  They also show examples of live-learning, as well as saving and loading weights.  The resulting saved weights are interchangeable between these two implementations.

Using the Neural Networks
---

To run the Octave/Matlab version, just run the script "nn.m" or call "nn" from an Octave/Matlab interpreter.  For C++, compile the runner file using "g++ main.cpp" and run "./a.out".  For Python, simply call Python with "runner.py" or use "execfile('runner.py')" from within a Python interpreter.  If Python reports problems, verify that numpy is installed.
