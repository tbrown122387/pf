# output binary
BIN := run_tests

# source files
SRCS := \
    run_tests.cpp test_resamplers.cpp test_rv_eval.cpp test_rv_samp.cpp 

# intermediate directory for generated object files
OBJDIR := .o
# intermediate directory for generated dependency files
DEPDIR := .d

# object files, auto generated from source files
OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SRCS)))

# compilers (at least gcc and clang) don't create the subdirectories automatically
$(shell mkdir -p $(DEPDIR))
$(shell mkdir -p $(dir $(OBJS)) >/dev/null)

# C++ compiler
CXX := g++
# linker
LD := g++

# C++ flags
CXXFLAGS := -std=c++11
# C/C++ flags
CPPFLAGS := -g -Wall -Wextra -pedantic -I/usr/local/include/UnitTest++ -I/usr/include/eigen3 -I../include
# linker flags
LDFLAGS := -no-pie "-L../bin" "-L/usr/local/lib"
# flags required for dependency generation; passed to compilers
DEPFLAGS = -MT $@ -MD -MP -MF $(DEPDIR)/$*.Td
# libraries
LDLIBS := -lUnitTest++

# compile C++ source files
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) -c -o $@
# link object files to binary
LINK.o = $(LD) $(LDFLAGS) -o $@ 
# precompile step
PRECOMPILE =
# postcompile step
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

all: $(BIN)

.PHONY: clean
clean:
	$(RM) -r $(OBJDIR) $(DEPDIR)

.PHONY: help
help:
	@echo available targets: all clean

$(BIN): $(OBJS)
	$(LINK.o) $^ $(LDLIBS)

$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	$(PRECOMPILE)
	$(COMPILE.cc) $<
	$(POSTCOMPILE)

.PRECIOUS = $(DEPDIR)/%.d
$(DEPDIR)/%.d: ;

-include $(DEPS)
