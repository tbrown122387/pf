PROG = examples
SRCS = main.cpp svol_comparison.cpp forward_sim.cpp resamp_comparison.cpp
OBJS = main.o svol_comparison.o forward_sim.o resamp_comparison.o
INCLUDES = -I../include -I/usr/include/eigen3
CXX = g++
CXXFLAGS = -std=c++11 -O3 $(INCLUDES) -pg


all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) -o $@ $(OBJS) 

main.cpp : svol_comparison.h forward_sim.h resamp_comparison.h
	$(CXX) $(CXXFLAGS) -c main.cpp 

#main.o : main.cpp svol_comparison.h forward_sim.h resamp_comparison.h
#	$(CXX) $(CXXFLAGS) -c main.cpp 

svol_comparison.o : svol_comparison.cpp svol_comparison.h svol_bs.h svol_apf.h svol_sisr.h 
	$(CXX) $(CXXFLAGS) -c svol_comparison.cpp  

forward_sim.o : forward_sim.cpp forward_sim.h svol_bs.h  
	$(CXX) $(CXXFLAGS) -c forward_sim.cpp  

resamp_comparison.o : resamp_comparison.cpp resamp_comparison.h svol_bs.h  
	$(CXX) $(CXXFLAGS) -c resamp_comparison.cpp  

.PHONY: clean
clean:
	rm -f $(PROG) $(OBJS) 

	

