#ifndef _TIMING_FIELD_
#define _TIMING_FIELD_

//Data type for field timing
typedef struct {
  int nmaps;
  int nside;
  int pol;
  double **maps;
  double *mask;
} field_data;

//Setup function for field timing
// - Initializes a set of maps and a mask
void *setup_field(int nside,int spin1,int spin2);

//Evaluator for field timing
// - Generates and frees an nmt_field object
void func_field(void *data);

//Destructor for field timing
void free_field(void *data);

#endif //_TIMING_FIELD_
