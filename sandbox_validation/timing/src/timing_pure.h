#ifndef _TIMING_PURE_
#define _TIMING_PURE_

//Data type for pure timing
typedef struct {
  int nside;
  double **maps;
  double *mask;
} pure_data;

//Setup function for pure timing
// - Initializes a set of maps and a mask
void *setup_pure(int nside,int spin1,int spin2);

//Evaluator for pure timing
// - Generates and frees an nmt_field object
void func_pure(void *data);

//Destructor for pure timing
void free_pure(void *data);

#endif //_TIMING_PURE_
