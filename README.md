# PF: a library for fast particle filtering!

[![DOI](https://zenodo.org/badge/130237492.svg)](https://zenodo.org/badge/latestdoi/130237492)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02599/status.svg)](https://doi.org/10.21105/joss.02599)

This is a template library for [particle filtering](https://en.wikipedia.org/wiki/Particle_filter). Templated abstract base classes for different particle filters are provided (e.g. the Bootstrap Filter, the SISR filter, the Auxiliary Particle Filter, the Rao-Blackwellized particle filter), as well as non-abstract (but indeed templated) base classes for closed-form filtering algorithms (e.g. Kalman Filter, Hidden Markov Model filter, etc.). 

Once you have a certain model in mind, all you have to do is make it into a class that inherits from the filter you want to use.

## Dependencies

This code makes use of the following libraries: 

- [Eigen v3.3](http://eigen.tuxfamily.org/) 
- [Boost v1.65.1](https://www.boost.org/)
- [Catch2 v2.9.2](https://github.com/catchorg/Catch2) 

Also, your compiler must enable C++17. 


## Installation

### Option 1: Install with `CMake` 

`git clone` this Github repostory, `cd` into the directory where everything is saved, then run the following commands:

    mkdir build && cd build/
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=/usr/local
    sudo cmake --build . --config Release --target install --parallel

You may subsitute another directory for `/usr/local`, if you wish. This will also build unit tests that can be run with the following command (assuming you're still in `build/`):

    ./test/pf_test

Note: for this to method work, you will need to install Catch2 "system wide" so that its `*.cmake` files are installed as well. To do this, [click here](https://github.com/catchorg/Catch2/blob/master/docs/cmake-integration.md#installing-catch2-from-git-repository). 

### Option 2: Drag-and-drop `.h` files

This is a header-only library, so there will be no extra building necessary. If you just want to copy the desired header files from `include/pf` into your own project, and build that project by itself, that's totally fine. There is no linking necessary, either. If you go this route, though, make sure to compile with C++17 enabled. Note, also, that this code all makes use of [Eigen v3.3](http://eigen.tuxfamily.org/) and [Boost v1.65.1](https://www.boost.org/). Unit tests use the [Catch2 v2.9.2](https://github.com/catchorg/Catch2) library.


## Examples

Don't know how to use this? No problem. Check out the [`examples`](https://github.com/tbrown122387/pf/tree/master/examples) sub-directory. This is a stand-alone cmake project, so you can just copy this sub-directory anywhere you like, and start editing.

For example, copy to `Desktop` and have at it:

    cp -r ~/pf/examples/ ~/Desktop/
    cd Desktop/examples/
    mkdir build && cd build
    cmake ..
    make

## Contributing

Want to contribute to this project? Great! Click [here](CONTRIBUTING.md) for details on how to do that.

## Paper

A full-length tutorial paper is available [here,](https://arxiv.org/abs/2001.10451) and a shorter introduction paper is available [here.](https://joss.theoj.org/papers/10.21105/joss.02599)

## Citation

Click the "DOI" link above. Or, if you're impatient, click ['here'](https://zenodo.org/record/2633289/export/hx) for a Bibtex citation.


