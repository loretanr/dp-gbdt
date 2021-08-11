COMPILER=g++
CFLAGS=-c -Wall -std=c++11 -g -O0# todo -O0 for hardening
MAKEFLAGS+="-j 8"
SOURCES=$(wildcard src/*.cpp src/gbdt/*.cpp)
INC=-I./include/ -I./include/gbdt/# -L ./src/ -lfmt

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

rebuild:
	make

# let the compiler do it's magic
fast:
	make CFLAGS="-c -Werror -std=c++11 -O3 -ffast-math -march=native -pthread"

profile:
	make CFLAGS="-g -pg -shared-libgcc -D TBB_USE_THREADING_TOOLS -c -Werror -std=c++11 -O3 -ffast-math -march=native -pthread" -j 4
