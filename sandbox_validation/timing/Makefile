CC = icc
OPTIONS = -Wall -openmp -O3
INC=-I/users/damonge/include -I../../src -I./src
LIB=-L/users/damonge/lib -L/usr/local/lib64
LIBALL=-lnmt -lfftw3 -lfftw3_omp -lgsl -lgslcblas -lcfitsio -lsharp -lfftpack -lc_utils -lm

UTILS = src/timing_utils.o
FIELD = src/timing_field.o
MCM = src/timing_mcm.o
PURE = src/timing_pure.o
DEPROJ = src/timing_deproj.o
PURE_DEPROJ = src/timing_pure_deproj.o
MAIN = src/main.c
OFILES = $(UTILS) $(FIELD) $(MCM) $(PURE) $(DEPROJ) $(PURE_DEPROJ)

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
