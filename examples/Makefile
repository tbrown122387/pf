PROG = examples
SRCS = main.cpp svol_comparison.cpp
OBJS = main.o svol_comparison.o
LIBS = -lpf 
INCLUDES = -I../include -I/usr/include/eigen3
CXX = g++
CXXFLAGS = -std=c++11 -O3 $(INCLUDES) -pg
LDFLAGS = "-L../bin"


all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

main.cpp : svol_comparison.h
	$(CXX) $(CXXFLAGS) -c 

main.o : main.cpp svol_comparison.h
	$(CXX) $(CXXFLAGS) -c main.cpp $(LDFLAGS) $(LIBS)

svol_comparison.o : svol_comparison.cpp svol_comparison.h svol_bs.h svol_apf.h svol_sisr.h 
	$(CXX) $(CXXFLAGS) -c svol_comparison.cpp $(LDFLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(PROG) $(OBJS) 

	

