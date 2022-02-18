# 2022 GSOC "Ideas List"

## The Big Picture

Particle filters are an indispensable class of algorithms for a wide variety of time series models. Unfortunately, 
  
  - they are complicated to describe and 
  - they are computationally intensive. 

The primary purpose of this library is to provide practitioners with the ***fastest and most easy-to-use code *** for particle filtering. All of our work/ideas will address one of these two difficulties. 

First, many statisticians and data scientists aren't familiar with compiled languages. That's why it's critical to provide a friendly interface for these folks, and our current examples could use some work. We need ***more examples, and these examples need to clearly demonstrate how to use compiled code from within an interpreted language.*** 

Second, we're looking to increase the ***speed, maintainability, and flexibility of this software***. As stated above, we want this to be the "go-to" software for particle filtering.


## Required Skills and Getting in Touch

All projects fall into one of two categories. Either you are either interested in 
  
  - the data analysis aspect of this project, or 
  - you are interested in software development. 

### Data Analysis Projects (The Big Picture)

If you are interested in doing data analysis projects with this software, you will need a strong background in statistics and data science. Experience with time series modeling would be helpful. So would experience with R or Python. 

If you're concerned about your lack of familiarity with a particular aspect of this project, please tell us. For instance, if you prefer interpreted languages such as R or Python, and are unsure about your ability to run c++ code, we can provide templates and examples, and/or pair you up with someone with complementary skills.

### Software Development Projects (The Big Picture)

If you are interested in software development, we'd love to talk to you about how you can help make us the "go-to" particle filtering library. 

This library is written in c++, so ideally that's your strongest language. Ideally, you will have at least some familiarity with `git`--that's what we use for version control. Experience with time series modeling and/or particle filtering is helpful but definitely not required.

If you would like to get involved, please drop us a line and describe which of the project ideas below sound most interesting to you. Be upfront about your strengths and weaknesses so that we can pair you up with the right people! If you have skills that span both categories, even better! Do let us know about that as well.

## 1. Statistical Case-Studies, Examples, Demonstrations and Improvements 

Primary skills involved: statistics, R/RStudio, or Python.
Category: Data Analysis
Secondary skills involved: basic c++.

  - Our current examples could be displayed more nicely, perhaps with RMarkdown, Jupyter notebooks, etc. Currently, the [`examples/`](https://github.com/tbrown122387/pf/tree/master/examples) directory is quite bare-bones and uninviting. While it could be used as a template for providing different examples, ***future examples need to be much more friendly.*** 
  - We need to provide examples in ***more subject areas.*** Currently, we are mostly focused on financial time series, and there is another example related to tracking. If you have heard that particle filters are useful in area "X", then this can probably be made into a new example.
  - We need to provide more examples of how to ***call this C++ code from within interpreted languages such as R and Python.*** This process is complicated by the templated nature of these classes.
  - We would also like to provide ***more particle filtering algorithms, resampling strategies, etc.*** We have a wide variety, but we can always implement new techniques from the literature. If you are a graduate student, and you're interested in writing good software for a method you're looking into, this topic could be right for you.

## 2. The Curiously Recurring Template Pattern

Primary skills involved: template metaprogramming.
Category: Software development
Secondary skills involved: general knowledge of time series analysis

Right now, this library's primary purpose is to provide abstract base classes. Each of these corresponds with a different type of particle filter. There are two issues we would like to tackle, and **we suspect CRTP might be the right tool for both jobs.***

First, as you can tell from the templated nature of this code, we are interested in performing as much work during compile time. CRTP might help us take this one step further if it can be used to convert our runtime polymorphism to compile-time polymorphism. 

Second, at the present moment, we have one base class for each particle filtering algorithm. These base classes require the user to implement pure virtual functions necessary for the algorithm to function. However, ***many different statistical models using the same particle filtering algorithm would benefit from having different signatures for these functions.*** The goal would be to create ***more succinct "base classes".*** We could derive many different versions of a given particle filter by inheriting from this more expressive class template. This would increase functionality for the end-user, and dramatically improve maintainability. 



## 3. Speed Ups, Unit Testing and Documentation 

Primary skills involved: mixed
Category: Software development
Minimum Difficulty: low

Everyone knows that attracting users is easier if 
they know our code is the fastest, and 
it is easy to get started using our code.
This category is concerned with timing code and improving unit testing and documentation.

 - There are probably a large number of places where speedups could be made. ***Everything is on the table here.***
 - Our Doxygen documentation could use some work. Many of the header files have outdated documentation!
 - Preliminary intra-library speed comparisons have been performed. They are quite favorable, but we could use a lot more. 
 - We have plenty of tests, but writing more is great for peace of mind. 
 - We currently do not use any automated testing or building tools. This opportunity category is wide open as well! 

