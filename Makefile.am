EXTRA_DIST = doc README.md

lib_LTLIBRARIES = libnmt.la
libnmt_la_SOURCES = src/utils.c src/flatsky_utils.c src/healpix_extra.c src/nmt_bins.c src/nmt_bins_flat.c src/nmt_field_flat.c src/nmt_mask.c src/nmt_mask_flat.c src/nmt_master.c src/nmt_covar.c src/nmt_master_flat.c src/nmt_covar_flat.c src/nmt_io.c
libnmt_la_CPPFLAGS = $(OPENMP_CFLAGS) -I./src/
libnmt_la_LDFLAGS = $(OPENMP_CFLAGS) @fftw_thread_type@ -lfftw3 -lgsl -lgslcblas -lcfitsio -lm

include_HEADERS = src/namaster.h

noinst_PROGRAMS = test/check_nmt
TESTS = test/check_nmt
test_check_nmt_LDADD = libnmt.la
test_check_nmt_SOURCES = test/nmt_test.c test/nmt_test_utils.c test/nmt_test_fsk.c test/nmt_test_hpex.c test/nmt_test_bins.c test/nmt_test_bins_flat.c test/nmt_test_field_flat.c test/nmt_test_master_flat.c test/nmt_test_covar_flat.c test/nmt_test_mask.c
test_check_nmt_CPPFLAGS = $(OPENMP_CFLAGS) -I./test/ -I./src/
test_check_nmt_LDFLAGS = $(OPENMP_CFLAGS) -L./ -lnmt @fftw_thread_type@ -lfftw3 -lgsl -lgslcblas -lcfitsio -lm
