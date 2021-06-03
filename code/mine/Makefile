COMPILER=g++
CFLAGS=-c -Wall -std=c++11 -g -O0
SOURCES=$(wildcard src/*.cpp)
INC=-I./include/

$(shell mkdir -p objs)

run: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) -pthread $^ -o $@

objs/%.o: src/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@
clean:
	rm -r objs
	rm run