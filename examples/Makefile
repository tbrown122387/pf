PROG = examples
SRCS = main.cpp svol_comparison.cpp
OBJS = main.o svol_comparison.o
LIBS = -lpf 

CXX = g++
CXXFLAGS = -std=c++11 -O3 $(INCLUDES) -pg
LDFLAGS = "-L/home/taylor/pf/bin"
INCLUDES = -I/home/taylor/pf/include -I/home/taylor/pf/examples -I/usr/include/eigen3

all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

.SUFFIXES: $(SUFFIXES) .cpp 
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

main.o : main.cpp svol_comparison.h
svol_comparison.o : svol_comparison.cpp svol_bs.h svol_apf.h svol_sisr.h 

.PHONY: clean
clean:
	rm -f $(PROG) $(OBJS) *.mod

	

