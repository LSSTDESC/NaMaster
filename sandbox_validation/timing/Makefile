CC = gcc
OPTIONS = -Wall -fopenmp
INC=-I/home/alonso/include -I../../src -I./src
LIB=-L/home/alonso/lib -L/usr/local/lib64
LIBALL=-lnmt -lfftw3 -lfftw3_omp -lgsl -lgslcblas -lcfitsio -lsharp -lfftpack -lc_utils -lm

UTILS = src/timing_utils.o
FIELD = src/timing_field.o
MCM = src/timing_mcm.o
PURE = src/timing_pure.o
MAIN = src/main.c
OFILES = $(UTILS) $(FIELD) $(MCM) $(PURE)

EXE = timings

default : $(EXE)

%.o : %.c
	$(CC) $(OPTIONS) $(INC) -c $< -o $@

$(EXE) : $(OFILES) $(MAIN)
	$(CC) $(OPTIONS) $(INC) $(OFILES) $(MAIN) -o $(EXE) $(LIB) $(LIBALL)

clean :
	rm -f src/*.o

cleaner :
	rm -f *~ src/*.o src/*~ $(EXE)
