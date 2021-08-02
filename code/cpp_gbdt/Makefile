COMPILER=g++
OBJDIR=objs/
SRCDIR=src/
CFLAGS=-c -Wall -std=c++11 -g -O0# todo -O0 for hardening
SOURCES=$(wildcard src/*.cpp)
INC=-I./include/# -L ./src/ -lfmt

BUILD_PRINT = \e[1;34mBuilding $<\e[0m

$(shell mkdir -p objs)

run: $(SOURCES:src/%.cpp=objs/%.o)
	$(COMPILER) -pthread $^ -o $@

clean:
	rm -r objs
	touch run && rm run

objs/%.o: src/%.cpp
	$(COMPILER) $(CFLAGS) $(INC) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
    @$(ECHO) "$< -> $@"
    @test -d $(OBJDIR) || mkdir -pm 775 $(OBJDIR)
    @test -d $(@D) || mkdir -pm 775 $(@D)
    @-$(RM) $@
    $(COMPILER) $(CFLAGS) $(INC) -c $< -o $@

# rebuild: clean 
# 	make

rebuild:
	make

# let the compiler do it's magic
fast:
	make CFLAGS="-c -Werror -std=c++11 -O3 -ffast-math -march=native -pthread"
	# -ftree-vectorize -fopt-info-vec
