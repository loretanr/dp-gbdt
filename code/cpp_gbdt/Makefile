COMPILER=g++
CFLAGS=-c -Wall -std=c++11 -g -O0# todo -O0 for hardening
SOURCES=$(wildcard src/*.cpp src/gbdt/*.cpp)
INC=-I./include/ -I./include/gbdt/# -L ./src/ -lfmt

BUILD_PRINT = \e[1;34mBuilding $<\e[0m

$(shell mkdir -p objs)
$(shell mkdir -p objs/gbdt)

run: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) -pthread $^ -o $@

clean:
	rm -r objs
	touch run && rm run

objs/%.o: src/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@
objs/gbdt/%.o: src/gbdt/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@


# rebuild: clean 
# 	make

rebuild:
	make

# let the compiler do it's magic
fast:
	make CFLAGS="-c -Werror -std=c++11 -O3 -ffast-math -march=native -pthread"
	# -ftree-vectorize -fopt-info-vec
