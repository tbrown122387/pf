# PF: a library for fast particle filtering!

This is a static library for fast particle filtering. Abstract base classes for different particle filters are provided (e.g. the Bootstrap Filter, the Auxiliary Particle Filter, Rao-Blackwellized particle filter, etc.), as well as non-abstract base classes for closed form filtering algorithms (e.g. Kalman Filter, Hidden Markov Model filter, etc.). iMost (all?) of these classes are templated for speed. Once you have a certain model in mind, make it into a class and inherit from an appropriate particle filter. 

## Installation
You must build this yourself. Make sure to compile with C++11 enabled (`-std=c++11`), and to include the `include` directory. Note, also, that this code all makes use of the [Eigen library](http://eigen.tuxfamily.org/).

## Examples
Don't know how to use this? Check out the `examples` directory!

## Documentation
[Documentation can be found here.](https://tbrown122387.github.io/pf/) 


