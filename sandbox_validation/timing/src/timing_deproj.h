#ifndef _TIMING_DEPROJ_
#define _TIMING_DEPROJ_

//Data type for deproj timing
typedef struct {
  int nmaps;
  int nside;
  int pol;
  double **maps;
  double ***temp;
  double *mask;
} deproj_data;

//Setup function for deproj timing
// - Initializes a set of maps and a mask
void *setup_deproj(int nside,int spin1,int spin2);

//Evaluator for deproj timing
// - Generates and frees an nmt_field object with templates
void func_deproj(void *data);

//Destructor for deproj timing
void free_deproj(void *data);

#endif //_TIMING_DEPROJ_
