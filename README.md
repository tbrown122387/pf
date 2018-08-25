# PF: a library for fast particle filtering!

This is a static library for fast particle filtering. Abstract base classes for different particle filters are provided (e.g. the Bootstrap Filter, the Auxiliary Particle Filter, Rao-Blackwellized particle filter, etc.), as well as non-abstract base classes for closed form filtering algorithms (e.g. Kalman Filter, Hidden Markov Model filter, etc.). Most (all?) of these classes are templated for speed. Once you have a certain model in mind, make it into a class and inherit from an appropriate particle filter. 

## Installation
Build this yourself using `build_lib.sh` (possibly editing the variable `EIGEN`). When you use it in another project, make sure to compile with C++11 enabled (`-std=c++11`), and to include the `include` directory of this project. Note, also, that this code all makes use of the [Eigen library](http://eigen.tuxfamily.org/).

## Examples
Don't know how to use this? Check out the [`examples`](https://github.com/tbrown122387/pf/tree/master/examples) directory! After you have built `libpf.a`, edit `pf/examples/Makefile` and then run `make`. After that, run `./examples ./data/svol_y_data.csv` and you'll see the filtering output from `examples/svol_comparison.cpp`.
