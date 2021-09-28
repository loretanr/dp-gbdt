COMPILER=g++
CFLAGS=-c -Wall -std=c++11 -g -O0 -I.
MAKEFLAGS+="-j 8"
DEPS = constant_time.h
INC=-I. -I./libfixedtime
LDFLAGS=-Wl,-V -L./libfixedtime --verbose

# "all cpp files depend on certain .h files"
%.o: %.cpp $(DEPS)
	$(COMPILER) $(CFLAGS) -fPIC $(INC) $< -o $@


run: main.o constant_time.o
	$(COMPILER) -pthread $(INC) $(LDFLAGS) -lftfp $^ -o $@ 



clean:
	rm *.o
	touch run && rm run