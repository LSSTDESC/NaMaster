EXTRA_DIST = doc README.md

lib_LTLIBRARIES = libnmt.la
libnmt_la_SOURCES = src/utils.c src/flatsky_utils.c src/healpix_extra.c src/nmt_bins.c src/nmt_bins_flat.c src/nmt_field_flat.c src/nmt_mask.c src/nmt_mask_flat.c src/nmt_master.c src/nmt_master_flat.c src/nmt_covar_flat.c
libnmt_la_CPPFLAGS = $(OPENMP_CFLAGS) -I./src/
libnmt_la_LDFLAGS = $(OPENMP_CFLAGS) @fftw_thread_type@ -lfftw3 -lgsl -lgslcblas -lcfitsio -lm

include_HEADERS = src/namaster.h
