#ifndef _TIMING_PURE_DEPROJ_
#define _TIMING_PURE_DEPROJ_

//Data type for pure_deproj timing
typedef struct {
  int nside;
  double **maps;
  double ***temp;
  double *mask;
} pure_deproj_data;

//Setup function for pure_deproj timing
// - Initializes a set of maps and a mask
void *setup_pure_deproj(int nside,int spin1,int spin2);

//Evaluator for pure_deproj timing
// - Generates and frees an nmt_field object
void func_pure_deproj(void *data);

//Destructor for pure_deproj timing
void free_pure_deproj(void *data);

#endif //_TIMING_PURE_DEPROJ_
