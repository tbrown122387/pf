# 2022 GSOC "Ideas List"

## The Big Picture

Particle filters are an indispensable class of algorithms for a wide variety of time series models. Unfortunately, they are complicated to describe and they are computationally intensive. The primary purpose of this library is to provide practitioners with the fastest and most easy to use particle filtering code. 

There are two difficulties, and all of our ideas will address one of these two. 

First, many statisticians and data scientists aren't familiar with compiled languages. That's why it's very important to provide an inviting interface for these folks. We need more examples, these examples need to clearly demonstrate how to use compiled code from within an interpreted language, and the examples we currently have need to be made more user-friendly. 

Second, it is important to build upon the quality of software. In particular, we're looking to increase speed, maintainability and flexibility.


## Required Skills and Getting in Touch

All projects fall into one of two categories. Either you are either interested in the data analysis aspect of this project, or you are interested in software development. 

If you are interested in getting involved with the first category of work, you would ideally possess a strong background in statistics and data science. Experience with time series modeling would be helpful too. So would experience with R or Python.

If you are interested in getting involved with the second category, you will need to possess strong skills in software development. Depending on the work you choose, that could mean different things. 

In either case, you will need to have at least some familiarity with `git` and `c++`. Experience with time series modeling and/or particle filtering is helpful but not required.

If you would like to get involved, please write to `TODO` and describe which of these areas you would like to work in, and how your previous experience would make you suitable for that work. 

## 1. Statistical Case-Studies, Examples, Demonstrations and Improvements 

Primary skills involved: statistics, R/RStudio or Python.
Category: Data Analysis
Minimum Difficulty: low

Secondary skills involved: basic c++.

  - Our current examples could be displayed more nicely, perhaps with RMarkdown, Jupyter notebooks, etc. Currently, the [`examples/`](https://github.com/tbrown122387/pf/tree/master/examples) directory is quite bare-bones and uninviting. This could be made more inviting, or it could be used as a template for providing different examples. 
  - We need to provide examples in more subject areas. Currently we are mostly focused on financial time series, and there is another example related to tracking. If you have heard that particle filters are useful in area X, then this can probably be made into a new example.
  - We need to provide more examples of how to call this C++ code from within interpreted languages such as R and Python. This process is complicated by the templated nature of these classes.
  - We would also like to provide more particle filtering algorithms, resampling strategies, etc. We have a wide variety, but we can always implement new techniques from the literature. If you are a graduate student, and you're interested in writing good software for a method you're looking into, this topic could be right for you.

## 2. The Curiously Recurring Template Pattern

Primary skills involved: template metaprogramming.
Category: Software development
Minimum Difficulty: high

Work in this category would probably be more ambitious than in any other category as it would require a significant amount of refactoring.

Right now, this library's primary purpose is to provide abstract base classes. Each of these corresponds with a different type of particle filter. There are two issues we would like to tackle, and we think CRTP might be the right tool for the job. 

First, as you can tell from the templated nature of this code, we are concerned with performing as much work during compile time. CRTP might be able to take us further if it can be used to convert our runtime polymorphism to compile time polymorphism. 

There is a second reason to leave behind Right now, we have one base class for each particle filtering algorithm. These base classes require the user to implement pure virtual functions that are necessary for the algorithm to function. However, many different statistical models that use the same particle filtering algorithm would benefit from having different signatures for these functions. The goal would be to create more succinct "base classes" that allow the user more flexibility in defining a statistical model, but would retain ease of use. 



## 3. Speed Ups, Unit Testing and Documentation 

Primary skills involved: mixed
Category: Software development
Minimum Difficulty: low


We can attract more users if they know our code is the fastest, and if they bother to figure out how to use it. 

 - There are probably a large number of places where speedups could be made. Everything is on the table here. 
 - Our Doxygen documentation could use some work. Many of the header files have outdated documentation.
 - Preliminary intra-library speed comparisons have been performed, but we could use more. 
 - We have plenty of tests, but writing more is always going to be good fo peace of mind. 
 - We currently do not use any automated testing or building tools. 

