EXTRA_DIST = doc README.md

lib_LTLIBRARIES = libnmt.la
libnmt_la_SOURCES = src/utils.c src/flatsky_utils.c src/healpix_extra.c src/nmt_bins.c src/nmt_bins_flat.c src/nmt_field.c src/nmt_field_flat.c src/nmt_mask.c src/nmt_mask_flat.c src/nmt_master.c src/nmt_covar.c src/nmt_master_flat.c src/nmt_covar_flat.c
libnmt_la_CPPFLAGS = $(OPENMP_CFLAGS) -I./src/

include_HEADERS = src/namaster.h

bin_PROGRAMS = namaster
namaster_SOURCES = src/nmt_main.c
namaster_CPPFLAGS = $(OPENMP_CFLAGS) -I./src/
namaster_LDFLAGS = $(OPENMP_CFLAGS) -L./ -lnmt -lfftw3 -lfftw3_omp -lgsl -lgslcblas -lcfitsio -lsharp -lfftpack -lc_utils -lm 

TESTS = test/nmt_checks
check_PROGRAMS = test/nmt_checks
test_nmt_checks_SOURCES = test/nmt_checks.c
test_nmt_checks_CPPFLAGS = $(OPENMP_CFLAGS) -I./src/ -I./test/
test_nmt_checks_LDFLAGS = $(OPENP_CFLAGS) -L./ -lnmt -lfftw3 -lfftw3_omp -lgsl -lgslcblas -lcfitsio -lsharp -lfftpack -lc_utils -lm 
