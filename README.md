# PF: a library for fast particle filtering!

[![DOI](https://zenodo.org/badge/130237492.svg)](https://zenodo.org/badge/latestdoi/130237492)

This is a template library for fast particle filtering. Templated abstract base classes for different particle filters are provided (e.g. the Bootstrap Filter, the Auxiliary Particle Filter, Rao-Blackwellized particle filter, etc.), as well as non-abstract (but indeed templated) base classes for closed form filtering algorithms (e.g. Kalman Filter, Hidden Markov Model filter, etc.). 

Once you have a certain model in mind, all you have to do is make it into a class that inherits from the filter you want to use!

## Installation
This is a header-only library, so there is no building necessary. When you use it in another project, make sure to compile with C++11 enabled (`-std=c++11`), and to include the `include` directory of this project. 

Note, also, that this code all makes use of [Eigen](http://eigen.tuxfamily.org/) and [Boost](https://www.boost.org/).

## Examples
Don't know how to use this? Check out the [`examples`](https://github.com/tbrown122387/pf/tree/master/examples) directory! Check `pf/examples/Makefile` to make sure it jives with your directories, and then run `make`. After that, run `./examples ./data/svol_y_data.csv` and you'll see the filtering output from `examples/svol_comparison.cpp`.

## Citation

Click the "DOI" link above. Or, if you're impatient, click ['here'](https://zenodo.org/record/2633289/export/hx) for a Bibtex citation.
