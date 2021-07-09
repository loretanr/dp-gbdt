COMPILER=g++
CFLAGS=-c -Wall -std=c++11 -g -O0
SOURCES=$(wildcard src/*.cpp)
INC=-I./include/ # -L ./src/ -lfmt

BUILD_PRINT = \e[1;34mBuilding $<\e[0m

$(shell mkdir -p objs)

run: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) -pthread $^ -o $@

clean:
	rm -r objs
	touch run && rm run

objs/%.o: src/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@

# rebuild: clean 
# 	make
rebuild:
	make