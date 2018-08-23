#ifndef _TIMING_MCM_
#define _TIMING_MCM_
//Data type for mcm timing
typedef struct {
  nmt_field *f1;
  nmt_field *f2;
  nmt_binning_scheme *bin;
} mcm_data;

//Setup function for field timing
// - Initializes a set of maps and a mask
void *setup_mcm(int nside,int spin1,int spin2);

//Evaluator for mcm timing
// - Generates and frees an nmt_workspace object
void func_mcm(void *data);

//Destructor for field timing
void free_mcm(void *data);

#endif //_TIMING_MCM_
