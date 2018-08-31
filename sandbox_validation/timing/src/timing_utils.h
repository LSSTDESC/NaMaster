#ifndef _TIMING_UTILS_
#define _TIMING_UTILS_

#define NTEMP 5

extern double relbeg,relend;
void timer(int i);

//Initializes a mask that is all zeros below the equator and one above
double *init_mask(int nside);
//Initializes a set of maps by populating them with random numbers
double **init_maps(int nside,int nmaps) ;
//Times a given function
double timing(int ncomp,int nside,int spin1,int spin2,
	      void *setup_time(int,int,int),void func_time(void *),void free_time(void *));

//Structure for task timing
typedef struct {
  char name[16];
  int nside;
  int nruns;
  void *(*setup)(int,int,int);
  void (*func)(void *);
  void (*free)(void *);
  double times[3];
} timing_st;

//timing_st constructor
timing_st *timing_st_init(char *name,int nside,int nruns,
			  void *(*setup_time)(int,int,int),
			  void (*func_time)(void *),
			  void (*free_time)(void *));

//timing_st destructor
void timing_st_free(timing_st *tim);

//timing_st timer
void timing_st_time(timing_st *tim,int ncomp);

//Report timings
void timing_st_report(timing_st *tim,FILE *f);


#endif //_TIMING_UTILS_
