#include <stdarg.h>

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <time.h>
#include <math.h>

#include <limits.h>
#include <cuda.h>

#ifndef NULL
#define NULL 0
#endif /* NULL */

#ifndef _WIN32
#include <sys/time.h>
#else /* _WIN32 */
#if !defined(_MSC_EXTENSIONS) && !defined(_INC_WINDOWS)
extern unsigned long __stdcall GetTickCount(void);

#else /* _MSC_EXTENSIONS */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif /* _MSC_EXTENSIONS */
#endif /* _WIN32 */

#if defined(_MSC_VER) && (_MSC_VER > 1300)
#ifndef FANN_NO_DLL
#define FANN_USE_DLL
#endif /* FANN_USE_LIB */
#endif /* _MSC_VER */
#if defined(_MSC_VER) && (defined(FANN_USE_DLL) || defined(FANN_DLL_EXPORTS))
#ifdef FANN_DLL_EXPORTS
#define FANN_EXTERNAL __declspec(dllexport)
#else /*  */
#define FANN_EXTERNAL __declspec(dllimport)
#endif /* FANN_DLL_EXPORTS*/
#define FANN_API __stdcall
#else /*  */
#define FANN_EXTERNAL
#define FANN_API
#endif /* _MSC_VER */

#define FANN_ERRSTR_MAX 128
typedef double fann_type;


struct fann_error;
struct fann_train_data;

enum fann_errno_enum {
  FANN_E_NO_ERROR = 0,
  FANN_E_CANT_OPEN_CONFIG_R,
  FANN_E_CANT_OPEN_CONFIG_W,
  FANN_E_WRONG_CONFIG_VERSION,
  FANN_E_CANT_READ_CONFIG,
  FANN_E_CANT_READ_NEURON,
  FANN_E_CANT_READ_CONNECTIONS,
  FANN_E_WRONG_NUM_CONNECTIONS,
  FANN_E_CANT_OPEN_TD_W,
  FANN_E_CANT_OPEN_TD_R,
  FANN_E_CANT_READ_TD,
  FANN_E_CANT_ALLOCATE_MEM,
  FANN_E_CANT_TRAIN_ACTIVATION,
  FANN_E_CANT_USE_ACTIVATION,
  FANN_E_TRAIN_DATA_MISMATCH,
  FANN_E_CANT_USE_TRAIN_ALG,
  FANN_E_TRAIN_DATA_SUBSET,
  FANN_E_INDEX_OUT_OF_BOUND,
  FANN_E_SCALE_NOT_PRESENT,
  FANN_E_INPUT_NO_MATCH,
  FANN_E_OUTPUT_NO_MATCH,
  FANN_E_WRONG_PARAMETERS_FOR_CREATE
};

struct fann_error {
  enum fann_errno_enum errno_f;
  FILE *error_log;
  char *errstr;
};

/* called fann_max, in order to not interferre with predefined versions of max */
#define fann_max(x, y) (((x) > (y)) ? (x) : (y))
#define fann_min(x, y) (((x) < (y)) ? (x) : (y))
#define fann_safe_free(x) \
  {                       \
    if (x) {              \
      free(x);            \
      x = NULL;           \
    }                     \
  }
#define fann_clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))
#define fann_exp2(x) exp(0.69314718055994530942 * (x))
/*#define fann_clip(x, lo, hi) (x)*/

#define fann_rand(min_value, max_value) \
  (((float)(min_value)) +               \
   (((float)(max_value) - ((float)(min_value))) * rand() / (RAND_MAX + 1.0f)))

#define fann_abs(value) (((value) > 0) ? (value) : -(value))


#define fann_mult(x, y) (x * y)
#define fann_div(x, y) (x / y)
#define fann_random_weight() (fann_rand(-0.1f, 0.1f))
#define fann_random_bias_weight() (fann_rand(-0.1f, 0.1f))



enum fann_train_enum {
  FANN_TRAIN_INCREMENTAL = 0,
  FANN_TRAIN_BATCH,
  FANN_TRAIN_RPROP,
  FANN_TRAIN_QUICKPROP,
  FANN_TRAIN_SARPROP
};

/* Constant: FANN_TRAIN_NAMES
   Constant array consisting of the names for the training algorithms, so that the name of an
   training function can be received by:
   (code)
   char *name = FANN_TRAIN_NAMES[train_function];
   (end)
   See Also:
      <fann_train_enum>
*/
static char const *const FANN_TRAIN_NAMES[] = {"FANN_TRAIN_INCREMENTAL", "FANN_TRAIN_BATCH",
                                               "FANN_TRAIN_RPROP", "FANN_TRAIN_QUICKPROP",
                                               "FANN_TRAIN_SARPROP"};

   
/* Group: Error Handling */

/* Function: fann_set_error_log
   Change where errors are logged to. Both <struct fann> and <struct fann_data> can be
   casted to <struct fann_error>, so this function can be used to set either of these.
   If log_file is NULL, no errors will be printed.
   If errdat is NULL, the default log will be set. The default log is the log used when creating
   <struct fann> and <struct fann_data>. This default log will also be the default for all new
   structs that are created.
   The default behavior is to log them to stderr.
   See also:
    <struct fann_error>
   This function appears in FANN >= 1.1.0.
 */
  void FANN_API fann_set_error_log(struct fann_error *errdat, FILE *log_file);

/* Function: fann_get_errno
   Returns the last error number.
   See also:
    <fann_errno_enum>, <fann_reset_errno>
   This function appears in FANN >= 1.1.0.
 */
  enum fann_errno_enum FANN_API fann_get_errno(struct fann_error *errdat);

/* Function: fann_reset_errno
   Resets the last error number.
   This function appears in FANN >= 1.1.0.
 */
  void FANN_API fann_reset_errno(struct fann_error *errdat);

/* Function: fann_reset_errstr
   Resets the last error string.
   This function appears in FANN >= 1.1.0.
 */
  void FANN_API fann_reset_errstr(struct fann_error *errdat);

/* Function: fann_get_errstr
   Returns the last errstr.
   This function calls <fann_reset_errno> and <fann_reset_errstr>
   This function appears in FANN >= 1.1.0.
 */
  char *FANN_API fann_get_errstr(struct fann_error *errdat);

/* Function: fann_print_error
   Prints the last error to stderr.
   This function appears in FANN >= 1.1.0.
 */
  void FANN_API fann_print_error(struct fann_error *errdat);

  extern FILE *fann_default_error_log;


/* resets the last error number
 */
    void FANN_API fann_reset_errno(struct fann_error *errdat) {
  errdat->errno_f = FANN_E_NO_ERROR;
}

/* resets the last errstr
 */
    void FANN_API fann_reset_errstr(struct fann_error *errdat) {
  if (errdat->errstr != NULL) free(errdat->errstr);
  errdat->errstr = NULL;
}

/* returns the last error number
 */
    enum fann_errno_enum FANN_API fann_get_errno(struct fann_error *errdat) {
  return errdat->errno_f;
}

/* returns the last errstr
 */
    char *FANN_API fann_get_errstr(struct fann_error *errdat) {
  char *errstr = errdat->errstr;

  fann_reset_errno(errdat);
  fann_reset_errstr(errdat);

  return errstr;
}


/* prints the last error to stderr
 */
    void FANN_API fann_print_error(struct fann_error *errdat) {
  if (errdat->errno_f != FANN_E_NO_ERROR && errdat->errstr != NULL) {
    fprintf(stderr, "FANN Error %d: %s", errdat->errno_f, errdat->errstr);
  }
}


#if defined(_WIN32) && !defined(__MINGW32__)
#define PATH_MAX _MAX_PATH
#endif
#ifndef PATH_MAX
#ifdef _POSIX_PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#else
#define PATH_MAX 4096
#endif
#endif

/* INTERNAL FUNCTION
   Populate the error information
 */

FILE *FANN_API fann_default_error_log = (FILE *)-1;

void fann_error(struct fann_error *errdat, const enum fann_errno_enum errno_f, ...) {
  va_list ap;
  size_t errstr_max = FANN_ERRSTR_MAX + PATH_MAX - 1;
  char errstr[FANN_ERRSTR_MAX + PATH_MAX];
  FILE *error_log = fann_default_error_log;

  if (errdat != NULL) errdat->errno_f = errno_f;

  va_start(ap, errno_f);
  switch (errno_f) {
    case FANN_E_NO_ERROR:
      return;
    case FANN_E_CANT_OPEN_CONFIG_R:
      vsnprintf(errstr, errstr_max, "Unable to open configuration file \"%s\" for reading.\n", ap);
      break;
    case FANN_E_CANT_OPEN_CONFIG_W:
      vsnprintf(errstr, errstr_max, "Unable to open configuration file \"%s\" for writing.\n", ap);
      break;
    case FANN_E_WRONG_CONFIG_VERSION:
      vsnprintf(
          errstr, errstr_max,
          "Wrong version of configuration file, aborting read of configuration file \"%s\".\n", ap);
      break;
    case FANN_E_CANT_READ_CONFIG:
      vsnprintf(errstr, errstr_max, "Error reading \"%s\" from configuration file \"%s\".\n", ap);
      break;
    case FANN_E_CANT_READ_NEURON:
      vsnprintf(errstr, errstr_max, "Error reading neuron info from configuration file \"%s\".\n",
                ap);
      break;
    case FANN_E_CANT_READ_CONNECTIONS:
      vsnprintf(errstr, errstr_max, "Error reading connections from configuration file \"%s\".\n",
                ap);
      break;
    case FANN_E_WRONG_NUM_CONNECTIONS:
      vsnprintf(errstr, errstr_max, "ERROR connections_so_far=%d, total_connections=%d\n", ap);
      break;
    case FANN_E_CANT_OPEN_TD_W:
      vsnprintf(errstr, errstr_max, "Unable to open train data file \"%s\" for writing.\n", ap);
      break;
    case FANN_E_CANT_OPEN_TD_R:
      vsnprintf(errstr, errstr_max, "Unable to open train data file \"%s\" for writing.\n", ap);
      break;
    case FANN_E_CANT_READ_TD:
      vsnprintf(errstr, errstr_max, "Error reading info from train data file \"%s\", line: %d.\n",
                ap);
      break;
    case FANN_E_CANT_ALLOCATE_MEM:
      strcpy(errstr, "Unable to allocate memory.\n");
      break;
    case FANN_E_CANT_TRAIN_ACTIVATION:
      strcpy(errstr, "Unable to train with the selected activation function.\n");
      break;
    case FANN_E_CANT_USE_ACTIVATION:
      strcpy(errstr, "Unable to use the selected activation function.\n");
      break;
    case FANN_E_TRAIN_DATA_MISMATCH:
      strcpy(errstr, "Training data must be of equivalent structure.\n");
      break;
    case FANN_E_CANT_USE_TRAIN_ALG:
      strcpy(errstr, "Unable to use the selected training algorithm.\n");
      break;
    case FANN_E_TRAIN_DATA_SUBSET:
      vsnprintf(errstr, errstr_max,
                "Subset from %d of length %d not valid in training set of length %d.\n", ap);
      break;
    case FANN_E_INDEX_OUT_OF_BOUND:
      vsnprintf(errstr, errstr_max, "Index %d is out of bound.\n", ap);
      break;
    case FANN_E_SCALE_NOT_PRESENT:
      strcpy(errstr, "Scaling parameters not present.\n");
      break;
    case FANN_E_INPUT_NO_MATCH:
      vsnprintf(errstr, errstr_max,
                "The number of input neurons in the ann (%d) and data (%d) don't match\n", ap);
      break;
    case FANN_E_OUTPUT_NO_MATCH:
      vsnprintf(errstr, errstr_max,
                "The number of output neurons in the ann (%d) and data (%d) don't match\n", ap);
      break;
    case FANN_E_WRONG_PARAMETERS_FOR_CREATE:
      strcpy(errstr,
             "The parameters for create_standard are wrong, either too few parameters provided or "
             "a negative/very high value provided.\n");
      break;
  }
  va_end(ap);

  if (errdat != NULL) {
    if (errdat->errstr == NULL) {
      errdat->errstr = (char *)malloc(strlen(errstr) + 1);
    } else if (strlen(errdat->errstr) < strlen(errstr)) {
      errdat->errstr = (char *)realloc(errdat->errstr, strlen(errstr) + 1);
    }
    /* allocation failed */
    if (errdat->errstr == NULL) {
      fprintf(stderr, "Unable to allocate memory.\n");
      return;
    }
    strcpy(errdat->errstr, errstr);
    error_log = errdat->error_log;
  }

  if (error_log == (FILE *)-1) /* This is the default behavior and will give stderr */
  {
    fprintf(stderr, "FANN Error %d: %s", errno_f, errstr);
  } else if (error_log != NULL) {
    fprintf(error_log, "FANN Error %d: %s", errno_f, errstr);
  }
}


      
/* INTERNAL FUNCTION
   Initialize an error data strcuture
 */
void fann_init_error_data(struct fann_error *errdat) {
  errdat->errstr = NULL;
  errdat->errno_f = FANN_E_NO_ERROR;
  errdat->error_log = fann_default_error_log;
}

enum fann_activationfunc_enum {
  FANN_LINEAR = 0,
  FANN_THRESHOLD,
  FANN_THRESHOLD_SYMMETRIC,
  FANN_SIGMOID,
  FANN_SIGMOID_STEPWISE,
  FANN_SIGMOID_SYMMETRIC,
  FANN_SIGMOID_SYMMETRIC_STEPWISE,
  FANN_GAUSSIAN,
  FANN_GAUSSIAN_SYMMETRIC,
  /* Stepwise linear approximation to gaussian.
   * Faster than gaussian but a bit less precise.
   * NOT implemented yet.
   */
  FANN_GAUSSIAN_STEPWISE,
  FANN_ELLIOT,
  FANN_ELLIOT_SYMMETRIC,
  FANN_LINEAR_PIECE,
  FANN_LINEAR_PIECE_SYMMETRIC,
  FANN_SIN_SYMMETRIC,
  FANN_COS_SYMMETRIC,
  FANN_SIN,
  FANN_COS
};

/* Constant: FANN_ACTIVATIONFUNC_NAMES
   Constant array consisting of the names for the activation function, so that the name of an
   activation function can be received by:
   (code)
   char *name = FANN_ACTIVATIONFUNC_NAMES[activation_function];
   (end)
   See Also:
      <fann_activationfunc_enum>
*/

#define FANN_EXP(x) exp(x)
#define FANN_SIN(x) sin(x)
#define FANN_COS(x) cos(x)

#define fann_linear_func(v1, r1, v2, r2, sum) \
  (((((r2) - (r1)) * ((sum) - (v1))) / ((v2) - (v1))) + (r1))
#define fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, min, max, sum)           \
  (sum < v5 ? (sum < v3 ? (sum < v2 ? (sum < v1 ? min : fann_linear_func(v1, r1, v2, r2, sum)) \
                                    : fann_linear_func(v2, r2, v3, r3, sum))                   \
                        : (sum < v4 ? fann_linear_func(v3, r3, v4, r4, sum)                    \
                                    : fann_linear_func(v4, r4, v5, r5, sum)))                  \
            : (sum < v6 ? fann_linear_func(v5, r5, v6, r6, sum) : max))

/* FANN_LINEAR */
/* #define fann_linear(steepness, sum) fann_mult(steepness, sum) */
#define fann_linear_derive(steepness, value) (steepness)

/* FANN_SIGMOID */
/* #define fann_sigmoid(steepness, sum) (1.0f/(1.0f + exp(-2.0f * steepness * sum))) */
#define fann_sigmoid_real(sum) (1.0f / (1.0f + FANN_EXP(-2.0f * sum)))
#define fann_sigmoid_derive(steepness, value) (2.0f * steepness * value * (1.0f - value))

/* FANN_SIGMOID_SYMMETRIC */
/* #define fann_sigmoid_symmetric(steepness, sum) (2.0f/(1.0f + exp(-2.0f * steepness * sum))
 * - 1.0f) */
#define fann_sigmoid_symmetric_real(sum) (2.0f / (1.0f + FANN_EXP(-2.0f * sum)) - 1.0f)
#define fann_sigmoid_symmetric_derive(steepness, value) steepness*(1.0f - (value * value))

/* FANN_GAUSSIAN */
/* #define fann_gaussian(steepness, sum) (exp(-sum * steepness * sum * steepness)) */
#define fann_gaussian_real(sum) (FANN_EXP(-sum * sum))
#define fann_gaussian_derive(steepness, value, sum) (-2.0f * sum * value * steepness * steepness)

/* FANN_GAUSSIAN_SYMMETRIC */
/* #define fann_gaussian_symmetric(steepness, sum) ((exp(-sum * steepness * sum *
 * steepness)*2.0)-1.0) */
#define fann_gaussian_symmetric_real(sum) ((FANN_EXP(-sum * sum) * 2.0f) - 1.0f)
#define fann_gaussian_symmetric_derive(steepness, value, sum) \
  (-2.0f * sum * (value + 1.0f) * steepness * steepness)

/* FANN_ELLIOT */
/* #define fann_elliot(steepness, sum) (((sum * steepness) / 2.0f) / (1.0f + fann_abs(sum *
 * steepness)) + 0.5f) */
#define fann_elliot_real(sum) (((sum) / 2.0f) / (1.0f + fann_abs(sum)) + 0.5f)
#define fann_elliot_derive(steepness, value, sum) \
  (steepness * 1.0f / (2.0f * (1.0f + fann_abs(sum)) * (1.0f + fann_abs(sum))))

/* FANN_ELLIOT_SYMMETRIC */
/* #define fann_elliot_symmetric(steepness, sum) ((sum * steepness) / (1.0f + fann_abs(sum *
 * steepness)))*/
#define fann_elliot_symmetric_real(sum) ((sum) / (1.0f + fann_abs(sum)))
#define fann_elliot_symmetric_derive(steepness, value, sum) \
  (steepness * 1.0f / ((1.0f + fann_abs(sum)) * (1.0f + fann_abs(sum))))

/* FANN_SIN_SYMMETRIC */
#define fann_sin_symmetric_real(sum) (FANN_SIN(sum))
#define fann_sin_symmetric_derive(steepness, sum) (steepness * cos(steepness * sum))

/* FANN_COS_SYMMETRIC */
#define fann_cos_symmetric_real(sum) (FANN_COS(sum))
#define fann_cos_symmetric_derive(steepness, sum) (steepness * -sin(steepness * sum))

/* FANN_SIN */
#define fann_sin_real(sum) (FANN_SIN(sum) / 2.0f + 0.5f)
#define fann_sin_derive(steepness, sum) (steepness * cos(steepness * sum) / 2.0f)

/* FANN_COS */
#define fann_cos_real(sum) (FANN_COS(sum) / 2.0f + 0.5f)
#define fann_cos_derive(steepness, sum) (steepness * -sin(steepness * sum) / 2.0f)

#define fann_activation_switch(activation_function, value, result)                                 \
  switch (activation_function) {                                                                   \
    case FANN_LINEAR:                                                                              \
      result = (fann_type)value;                                                                   \
      break;                                                                                       \
    case FANN_LINEAR_PIECE:                                                                        \
      result = (fann_type)((value < 0) ? 0 : (value > 1) ? 1 : value);                             \
      break;                                                                                       \
    case FANN_LINEAR_PIECE_SYMMETRIC:                                                              \
      result = (fann_type)((value < -1) ? -1 : (value > 1) ? 1 : value);                           \
      break;                                                                                       \
    case FANN_SIGMOID:                                                                             \
      result = (fann_type)fann_sigmoid_real(value);                                                \
      break;                                                                                       \
    case FANN_SIGMOID_SYMMETRIC:                                                                   \
      result = (fann_type)fann_sigmoid_symmetric_real(value);                                      \
      break;                                                                                       \
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:                                                          \
      result = (fann_type)fann_stepwise(                                                           \
          ((fann_type)-2.64665293693542480469e+00), ((fann_type)-1.47221934795379638672e+00),      \
          ((fann_type)-5.49306154251098632812e-01), ((fann_type)5.49306154251098632812e-01),       \
          ((fann_type)1.47221934795379638672e+00), ((fann_type)2.64665293693542480469e+00),        \
          ((fann_type)-9.90000009536743164062e-01), ((fann_type)-8.99999976158142089844e-01),      \
          ((fann_type)-5.00000000000000000000e-01), ((fann_type)5.00000000000000000000e-01),       \
          ((fann_type)8.99999976158142089844e-01), ((fann_type)9.90000009536743164062e-01), -1, 1, \
          value);                                                                                  \
      break;                                                                                       \
    case FANN_SIGMOID_STEPWISE:                                                                    \
      result = (fann_type)fann_stepwise(                                                           \
          ((fann_type)-2.64665246009826660156e+00), ((fann_type)-1.47221946716308593750e+00),      \
          ((fann_type)-5.49306154251098632812e-01), ((fann_type)5.49306154251098632812e-01),       \
          ((fann_type)1.47221934795379638672e+00), ((fann_type)2.64665293693542480469e+00),        \
          ((fann_type)4.99999988824129104614e-03), ((fann_type)5.00000007450580596924e-02),        \
          ((fann_type)2.50000000000000000000e-01), ((fann_type)7.50000000000000000000e-01),        \
          ((fann_type)9.49999988079071044922e-01), ((fann_type)9.95000004768371582031e-01), 0, 1,  \
          value);                                                                                  \
      break;                                                                                       \
    case FANN_THRESHOLD:                                                                           \
      result = (fann_type)((value < 0) ? 0 : 1);                                                   \
      break;                                                                                       \
    case FANN_THRESHOLD_SYMMETRIC:                                                                 \
      result = (fann_type)((value < 0) ? -1 : 1);                                                  \
      break;                                                                                       \
    case FANN_GAUSSIAN:                                                                            \
      result = (fann_type)fann_gaussian_real(value);                                               \
      break;                                                                                       \
    case FANN_GAUSSIAN_SYMMETRIC:                                                                  \
      result = (fann_type)fann_gaussian_symmetric_real(value);                                     \
      break;                                                                                       \
    case FANN_ELLIOT:                                                                              \
      result = (fann_type)fann_elliot_real(value);                                                 \
      break;                                                                                       \
    case FANN_ELLIOT_SYMMETRIC:                                                                    \
      result = (fann_type)fann_elliot_symmetric_real(value);                                       \
      break;                                                                                       \
    case FANN_SIN_SYMMETRIC:                                                                       \
      result = (fann_type)fann_sin_symmetric_real(value);                                          \
      break;                                                                                       \
    case FANN_COS_SYMMETRIC:                                                                       \
      result = (fann_type)fann_cos_symmetric_real(value);                                          \
      break;                                                                                       \
    case FANN_SIN:                                                                                 \
      result = (fann_type)fann_sin_real(value);                                                    \
      break;                                                                                       \
    case FANN_COS:                                                                                 \
      result = (fann_type)fann_cos_real(value);                                                    \
      break;                                                                                       \
    case FANN_GAUSSIAN_STEPWISE:                                                                   \
      result = 0;                                                                                  \
      break;                                                                                       \
  }

/* Enum: fann_errorfunc_enum
        Error function used during training.
        FANN_ERRORFUNC_LINEAR - Standard linear error function.
        FANN_ERRORFUNC_TANH - Tanh error function, usually better
                but can require a lower learning rate. This error function aggressively targets
   outputs that differ much from the desired, while not targeting outputs that only differ a little
   that much. This activation function is not recommended for cascade training and incremental
   training.
        See also:
                <fann_set_train_error_function>, <fann_get_train_error_function>
*/


enum fann_errorfunc_enum { FANN_ERRORFUNC_LINEAR = 0, FANN_ERRORFUNC_TANH };

/* Constant: FANN_ERRORFUNC_NAMES
   Constant array consisting of the names for the training error functions, so that the name of an
   error function can be received by:
   (code)
   char *name = FANN_ERRORFUNC_NAMES[error_function];
   (end)
   See Also:
      <fann_errorfunc_enum>
*/
static char const *const FANN_ERRORFUNC_NAMES[] = {"FANN_ERRORFUNC_LINEAR", "FANN_ERRORFUNC_TANH"};

/* Enum: fann_stopfunc_enum
        Stop criteria used during training.
        FANN_STOPFUNC_MSE - Stop criterion is Mean Square Error (MSE) value.
        FANN_STOPFUNC_BIT - Stop criterion is number of bits that fail. The number of bits; means
   the number of output neurons which differ more than the bit fail limit (see
   <fann_get_bit_fail_limit>, <fann_set_bit_fail_limit>). The bits are counted in all of the
   training data, so this number can be higher than the number of training data.
        See also:
                <fann_set_train_stop_function>, <fann_get_train_stop_function>
*/
enum fann_stopfunc_enum { FANN_STOPFUNC_MSE = 0, FANN_STOPFUNC_BIT };

/* Constant: FANN_STOPFUNC_NAMES
   Constant array consisting of the names for the training stop functions, so that the name of a
   stop function can be received by:
   (code)
   char *name = FANN_STOPFUNC_NAMES[stop_function];
   (end)
   See Also:
      <fann_stopfunc_enum>
*/
static char const *const FANN_STOPFUNC_NAMES[] = {"FANN_STOPFUNC_MSE", "FANN_STOPFUNC_BIT"};

/* Enum: fann_network_type_enum
    Definition of network types used by <fann_get_network_type>
    FANN_NETTYPE_LAYER - Each layer only has connections to the next layer
    FANN_NETTYPE_SHORTCUT - Each layer has connections to all following layers
   See Also:
      <fann_get_network_type>
   This enumeration appears in FANN >= 2.1.0
*/
enum fann_nettype_enum {
  FANN_NETTYPE_LAYER = 0, /* Each layer only has connections to the next layer */
  FANN_NETTYPE_SHORTCUT   /* Each layer has connections to all following layers */
};

/* Constant: FANN_NETWORK_TYPE_NAMES
   Constant array consisting of the names for the network types, so that the name of an
   network type can be received by:
   (code)
   char *network_type_name = FANN_NETWORK_TYPE_NAMES[fann_get_network_type(ann)];
   (end)
   See Also:
      <fann_get_network_type>
   This constant appears in FANN >= 2.1.0
*/
static char const *const FANN_NETTYPE_NAMES[] = {"FANN_NETTYPE_LAYER", "FANN_NETTYPE_SHORTCUT"};

/* forward declarations for use with the callback */
struct fann;


struct fann_train_data {
  enum fann_errno_enum errno_f;
  FILE *error_log;
  char *errstr;

  unsigned int num_data;
  unsigned int num_input;
  unsigned int num_output;
  fann_type **input;
  fann_type **output;
};
/* Type: fann_callback_type
   This callback function can be called during training when using <fann_train_on_data>,
   <fann_train_on_file> or <fann_cascadetrain_on_data>.
        >typedef int (FANN_API * fann_callback_type) (struct fann *ann, struct fann_train_data
   *train,
        > unsigned int max_epochs, >                                             unsigned int
   epochs_between_reports, >                                             float desired_error,
   unsigned int epochs);
        The callback can be set by using <fann_set_callback> and is very useful for doing custom
        things during training. It is recommended to use this function when implementing custom
        training procedures, or when visualizing the training in a GUI etc. The parameters which the
        callback function takes are the parameters given to <fann_train_on_data>, plus an epochs
        parameter which tells how many epochs the training has taken so far.
        The callback function should return an integer, if the callback function returns -1, the
   training will terminate.
        Example of a callback function:
                >int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
                >				            unsigned int max_epochs, unsigned int
   epochs_between_reports,
                >				            float desired_error, unsigned int
   epochs)
                >{
                >	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs,
   fann_get_MSE(ann), desired_error); >	return 0;
                >}
        See also:
                <fann_set_callback>, <fann_train_on_data>
 */
  typedef int(FANN_API *fann_callback_type)(struct fann *ann,
                                                        struct fann_train_data *train,
                                                        unsigned int max_epochs,
                                                        unsigned int epochs_between_reports,
                                                        float desired_error, unsigned int epochs);

/* ----- Data structures -----
 * No data within these structures should be altered directly by the user.
 */

struct fann_neuron {
  /* Index to the first and last connection
   * (actually the last is a past end index)
   */
  unsigned int first_con;
  unsigned int last_con;
  /* The sum of the inputs multiplied with the weights */
  fann_type sum;
  /* The value of the activation function applied to the sum */
  fann_type value;
  /* The steepness of the activation function */
  fann_type activation_steepness;
  /* Used to choose which activation function to use */
  enum fann_activationfunc_enum activation_function;

};


/* A single layer in the neural network.
 */
struct fann_layer {
  /* A pointer to the first neuron in the layer
   * When allocated, all the neurons in all the layers are actually
   * in one long array, this is because we want to easily clear all
   * the neurons at once.
   */
  struct fann_neuron *first_neuron;

  /* A pointer to the neuron past the last neuron in the layer */
  /* the number of neurons is last_neuron - first_neuron */
  struct fann_neuron *last_neuron;
};

/* Struct: struct fann_error
        Structure used to store error-related information, both
        <struct fann> and <struct fann_train_data> can be casted to this type.
        See also:
                <fann_set_error_log>, <fann_get_errno>
*/


/* 	Struct: struct fann
        The fast artificial neural network (fann) structure.
        Data within this structure should never be accessed directly, but only by using the
        *fann_get_...* and *fann_set_...* functions.
        The fann structure is created using one of the *fann_create_...* functions and each of
        the functions which operates on the structure takes *struct fann * ann* as the first
   parameter.
        See also:
                <fann_create_standard>, <fann_destroy>
 */
struct fann {
  /* The type of error that last occured. */
  enum fann_errno_enum errno_f;

  /* Where to log error messages. */
  FILE *error_log;

  /* A string representation of the last error. */
  char *errstr;

  /* the learning rate of the network */
  float learning_rate;

  /* The learning momentum used for backpropagation algorithm. */
  float learning_momentum;

  /* the connection rate of the network
   * between 0 and 1, 1 meaning fully connected
   */
  float connection_rate;

  /* is 1 if shortcut connections are used in the ann otherwise 0
   * Shortcut connections are connections that skip layers.
   * A fully connected ann with shortcut connections are a ann where
   * neurons have connections to all neurons in all later layers.
   */
  enum fann_nettype_enum network_type;

  /* pointer to the first layer (input layer) in an array af all the layers,
   * including the input and outputlayers
   */
  struct fann_layer *first_layer;

  /* pointer to the layer past the last layer in an array af all the layers,
   * including the input and outputlayers
   */
  struct fann_layer *last_layer;

  /* Total number of neurons.
   * very useful, because the actual neurons are allocated in one long array
   */
  unsigned int total_neurons;

  /* Number of input neurons (not calculating bias) */
  unsigned int num_input;

  /* Number of output neurons (not calculating bias) */
  unsigned int num_output;

  /* The weight array */
  fann_type *weights;

  /* The connection array */
  struct fann_neuron **connections;

  /* Used to contain the errors used during training
   * Is allocated during first training session,
   * which means that if we do not train, it is never allocated.
   */
  fann_type *train_errors;

  /* Training algorithm used when calling fann_train_on_..
   */
  enum fann_train_enum training_algorithm;

  /* Total number of connections.
   * very useful, because the actual connections
   * are allocated in one long array
   */
  unsigned int total_connections;

  /* used to store outputs in */
  fann_type *output;

  /* the number of data used to calculate the mean square error.
   */
  unsigned int num_MSE;

  /* the total error value.
   * the real mean square error is MSE_value/num_MSE
   */
  float MSE_value;

  /* The number of outputs which would fail (only valid for classification problems)
   */
  unsigned int num_bit_fail;

  /* The maximum difference between the actual output and the expected output
   * which is accepted when counting the bit fails.
   * This difference is multiplied by two when dealing with symmetric activation functions,
   * so that symmetric and not symmetric activation functions can use the same limit.
   */
  fann_type bit_fail_limit;

  /* The error function used during training. (default FANN_ERRORFUNC_TANH)
   */
  enum fann_errorfunc_enum train_error_function;

  /* The stop function used during training. (default FANN_STOPFUNC_MSE)
   */
  enum fann_stopfunc_enum train_stop_function;

  /* The callback function used during training. (default NULL)
   */
  fann_callback_type callback;

  /* A pointer to user defined data. (default NULL)
   */
  void *user_data;

  /* Variables for use with Cascade Correlation */

  /* The error must change by at least this
   * fraction of its old value to count as a
   * significant change.
   */
  float cascade_output_change_fraction;

  /* No change in this number of epochs will cause
   * stagnation.
   */
  unsigned int cascade_output_stagnation_epochs;

  /* The error must change by at least this
   * fraction of its old value to count as a
   * significant change.
   */
  float cascade_candidate_change_fraction;

  /* No change in this number of epochs will cause
   * stagnation.
   */
  unsigned int cascade_candidate_stagnation_epochs;

  /* The current best candidate, which will be installed.
   */
  unsigned int cascade_best_candidate;

  /* The upper limit for a candidate score
   */
  fann_type cascade_candidate_limit;

  /* Scale of copied candidate output weights
   */
  fann_type cascade_weight_multiplier;

  /* Maximum epochs to train the output neurons during cascade training
   */
  unsigned int cascade_max_out_epochs;

  /* Maximum epochs to train the candidate neurons during cascade training
   */
  unsigned int cascade_max_cand_epochs;

  /* Minimum epochs to train the output neurons during cascade training
   */
  unsigned int cascade_min_out_epochs;

  /* Minimum epochs to train the candidate neurons during cascade training
   */
  unsigned int cascade_min_cand_epochs;

  /* An array consisting of the activation functions used when doing
   * cascade training.
   */
  enum fann_activationfunc_enum *cascade_activation_functions;

  /* The number of elements in the cascade_activation_functions array.
   */
  unsigned int cascade_activation_functions_count;

  /* An array consisting of the steepnesses used during cascade training.
   */
  fann_type *cascade_activation_steepnesses;

  /* The number of elements in the cascade_activation_steepnesses array.
   */
  unsigned int cascade_activation_steepnesses_count;

  /* The number of candidates of each type that will be present.
   * The actual number of candidates is then
   * cascade_activation_functions_count *
   * cascade_activation_steepnesses_count *
   * cascade_num_candidate_groups
   */
  unsigned int cascade_num_candidate_groups;

  /* An array consisting of the score of the individual candidates,
   * which is used to decide which candidate is the best
   */
  fann_type *cascade_candidate_scores;

  /* The number of allocated neurons during cascade correlation algorithms.
   * This number might be higher than the actual number of neurons to avoid
   * allocating new space too often.
   */
  unsigned int total_neurons_allocated;

  /* The number of allocated connections during cascade correlation algorithms.
   * This number might be higher than the actual number of neurons to avoid
   * allocating new space too often.
   */
  unsigned int total_connections_allocated;

  /* Variables for use with Quickprop training */

  /* Decay is used to make the weights not go so high */
  float quickprop_decay;

  /* Mu is a factor used to increase and decrease the stepsize */
  float quickprop_mu;

  /* Variables for use with with RPROP training */

  /* Tells how much the stepsize should increase during learning */
  float rprop_increase_factor;

  /* Tells how much the stepsize should decrease during learning */
  float rprop_decrease_factor;

  /* The minimum stepsize */
  float rprop_delta_min;

  /* The maximum stepsize */
  float rprop_delta_max;

  /* The initial stepsize */
  float rprop_delta_zero;

  /* Defines how much the weights are constrained to smaller values at the beginning */
  float sarprop_weight_decay_shift;

  /* Decides if the stepsize is too big with regard to the error */
  float sarprop_step_error_threshold_factor;

  /* Defines how much the stepsize is influenced by the error */
  float sarprop_step_error_shift;

  /* Defines how much the epoch influences weight decay and noise */
  float sarprop_temperature;

  /* Current training epoch */
  unsigned int sarprop_epoch;

  /* Used to contain the slope errors used during batch training
   * Is allocated during first training session,
   * which means that if we do not train, it is never allocated.
   */
  fann_type *train_slopes;

  /* The previous step taken by the quickprop/rprop procedures.
   * Not allocated if not used.
   */
  fann_type *prev_steps;

  /* The slope values used by the quickprop/rprop procedures.
   * Not allocated if not used.
   */
  fann_type *prev_train_slopes;

  /* The last delta applied to a connection weight.
   * This is used for the momentum term in the backpropagation algorithm.
   * Not allocated if not used.
   */
  fann_type *prev_weights_deltas;

};

/* Type: fann_connection
    Describes a connection between two neurons and its weight
    from_neuron - Unique number used to identify source neuron
    to_neuron - Unique number used to identify destination neuron
    weight - The numerical value of the weight
    See Also:
        <fann_get_connection_array>, <fann_set_weight_array>
   This structure appears in FANN >= 2.1.0
*/
struct fann_connection {
  /* Unique number used to identify source neuron */
  unsigned int from_neuron;
  /* Unique number used to identify destination neuron */
  unsigned int to_neuron;
  /* The numerical value of the weight */
  fann_type weight;
};

float FANN_API fann_get_MSE(struct fann *ann);

fann_type fann_update_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type neuron_diff);

struct fann *fann_allocate_structure(unsigned int num_layers);
void fann_allocate_neurons(struct fann *ann);

void fann_allocate_connections(struct fann *ann);

int fann_save_internal(struct fann *ann, const char *configuration_file,
                       unsigned int save_as_fixed);
int fann_save_internal_fd(struct fann *ann, FILE *conf, const char *configuration_file,
                          unsigned int save_as_fixed);
int fann_save_train_internal(struct fann_train_data *data, const char *filename,
                             unsigned int save_as_fixed, unsigned int decimal_point);
int fann_save_train_internal_fd(struct fann_train_data *data, FILE *file, const char *filename,
                                unsigned int save_as_fixed, unsigned int decimal_point);

void fann_update_stepwise(struct fann *ann);
void fann_seed_rand();

void fann_error(struct fann_error *errdat, const enum fann_errno_enum errno_f, ...);
void fann_init_error_data(struct fann_error *errdat);

struct fann *fann_create_from_fd(FILE *conf, const char *configuration_file);
struct fann_train_data *fann_read_train_from_fd(FILE *file, const char *filename);

void fann_compute_MSE(struct fann *ann, fann_type *desired_output);
void fann_update_output_weights(struct fann *ann);
void fann_backpropagate_MSE(struct fann *ann);
void fann_update_weights(struct fann *ann);
void fann_update_slopes_batch(struct fann *ann, struct fann_layer *layer_begin,
                              struct fann_layer *layer_end);
void fann_update_weights_quickprop(struct fann *ann, unsigned int num_data,
                                   unsigned int first_weight, unsigned int past_end);
void fann_update_weights_batch(struct fann *ann, unsigned int num_data, unsigned int first_weight,
                               unsigned int past_end);
void fann_update_weights_irpropm(struct fann *ann, unsigned int first_weight,
                                 unsigned int past_end);
void fann_update_weights_sarprop(struct fann *ann, unsigned int epoch, unsigned int first_weight,
                                 unsigned int past_end);

void fann_clear_train_arrays(struct fann *ann);

fann_type fann_activation(struct fann *ann, unsigned int activation_function, fann_type steepness,
                          fann_type value);

fann_type fann_activation_derived(unsigned int activation_function, fann_type steepness,
                                  fann_type value, fann_type sum);

int fann_desired_error_reached(struct fann *ann, float desired_error);

/* Some functions for cascade */
int fann_train_outputs(struct fann *ann, struct fann_train_data *data, float desired_error);

float fann_train_outputs_epoch(struct fann *ann, struct fann_train_data *data);

int fann_train_candidates(struct fann *ann, struct fann_train_data *data);

fann_type fann_train_candidates_epoch(struct fann *ann, struct fann_train_data *data);

void fann_install_candidate(struct fann *ann);
int fann_check_input_output_sizes(struct fann *ann, struct fann_train_data *data);

int fann_initialize_candidates(struct fann *ann);

void fann_set_shortcut_connections(struct fann *ann);

int fann_allocate_scale(struct fann *ann);

__global__ void Add_Mult_update_wts(fann_type* A, fann_type B, fann_type* C, const float constant, int N, fann_neuron* D);

void FANN_API fann_scale_data_to_range(fann_type **data, unsigned int num_data,
                                                     unsigned int num_elem, fann_type old_min,
                                                     fann_type old_max, fann_type new_min,
                                                     fann_type new_max);

void fann_compute_MSE(struct fann *ann, fann_type *desired_output) {
  fann_type neuron_value, neuron_diff, *error_it = 0, *error_begin = 0;
  struct fann_neuron *last_layer_begin = (ann->last_layer - 1)->first_neuron;
  const struct fann_neuron *last_layer_end = last_layer_begin + ann->num_output;
  const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

  /* if no room allocated for the error variabels, allocate it now */
  if (ann->train_errors == NULL) {
    ann->train_errors = (fann_type *)calloc(ann->total_neurons, sizeof(fann_type));
    if (ann->train_errors == NULL) {
      fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  } else {
    /* clear the error variabels */
    memset(ann->train_errors, 0, (ann->total_neurons) * sizeof(fann_type));
  }
  error_begin = ann->train_errors;

  /* calculate the error and place it in the output layer */
  error_it = error_begin + (last_layer_begin - first_neuron);

  for (; last_layer_begin != last_layer_end; last_layer_begin++) {
    neuron_value = last_layer_begin->value;
    neuron_diff = *desired_output - neuron_value;

    neuron_diff = fann_update_MSE(ann, last_layer_begin, neuron_diff);

    if (ann->train_error_function) { /* TODO make switch when more functions */
      if (neuron_diff < -.9999999)
        neuron_diff = -17.0;
      else if (neuron_diff > .9999999)
        neuron_diff = 17.0;
      else
        neuron_diff = (fann_type)log((1.0 + neuron_diff) / (1.0 - neuron_diff));
    }

    *error_it = fann_activation_derived(last_layer_begin->activation_function,
                                        last_layer_begin->activation_steepness, neuron_value,
                                        last_layer_begin->sum) *
                neuron_diff;

    desired_output++;
    error_it++;

    ann->num_MSE++;
  }
}

fann_type fann_activation_derived(unsigned int activation_function, fann_type steepness,
                                  fann_type value, fann_type sum) {
  switch (activation_function) {
    case FANN_LINEAR:
    case FANN_LINEAR_PIECE:
    case FANN_LINEAR_PIECE_SYMMETRIC:
      return (fann_type)fann_linear_derive(steepness, value);
    case FANN_SIGMOID:
    case FANN_SIGMOID_STEPWISE:
      value = fann_clip(value, 0.01f, 0.99f);
      return (fann_type)fann_sigmoid_derive(steepness, value);
    case FANN_SIGMOID_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
      value = fann_clip(value, -0.98f, 0.98f);
      return (fann_type)fann_sigmoid_symmetric_derive(steepness, value);
    case FANN_GAUSSIAN:
      /* value = fann_clip(value, 0.01f, 0.99f); */
      return (fann_type)fann_gaussian_derive(steepness, value, sum);
    case FANN_GAUSSIAN_SYMMETRIC:
      /* value = fann_clip(value, -0.98f, 0.98f); */
      return (fann_type)fann_gaussian_symmetric_derive(steepness, value, sum);
    case FANN_ELLIOT:
      value = fann_clip(value, 0.01f, 0.99f);
      return (fann_type)fann_elliot_derive(steepness, value, sum);
    case FANN_ELLIOT_SYMMETRIC:
      value = fann_clip(value, -0.98f, 0.98f);
      return (fann_type)fann_elliot_symmetric_derive(steepness, value, sum);
    case FANN_SIN_SYMMETRIC:
      return (fann_type)fann_sin_symmetric_derive(steepness, sum);
    case FANN_COS_SYMMETRIC:
      return (fann_type)fann_cos_symmetric_derive(steepness, sum);
    case FANN_SIN:
      return (fann_type)fann_sin_derive(steepness, sum);
    case FANN_COS:
      return (fann_type)fann_cos_derive(steepness, sum);
    case FANN_THRESHOLD:
      fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
  }
  return 0;
}
/* INTERNAL FUNCTION
   Propagate the error backwards from the output layer.
   After this the train_errors in the hidden layers will be:
   neuron_value_derived * sum(outgoing_weights * connected_neuron)
*/
__global__ void backpropagate_gpu(int N, fann_type* output, fann_type temp, fann_type* input)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i <= N)
    {
      output[N-i]+=temp*input[N-i];
    }
}


void fann_backpropagate_MSE(struct fann *ann) {
  
  fann_type tmp_error;
  unsigned int i;
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it, *last_neuron;
  struct fann_neuron **connections;

  fann_type *error_begin = ann->train_errors;
  fann_type *error_prev_layer;
  fann_type *weights;
  const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  const struct fann_layer *second_layer = ann->first_layer + 1;
  struct fann_layer *last_layer = ann->last_layer;
  
  // threads, blocks

  int threadsperblock;
  int numBlocks;
  int N;

  //device parameters
  // device parameters
  fann_type* d_errors;
  fann_type* d_wts;

  threadsperblock=256;
  
  /* go through all the layers, from last to first.
   * And propagate the error backwards */
  for (layer_it = last_layer - 1; layer_it > second_layer; --layer_it) {
    last_neuron = layer_it->last_neuron;

    /* for each connection in this layer, propagate the error backwards */
    if (ann->connection_rate >= 1) {
      if (ann->network_type == FANN_NETTYPE_LAYER) {
        error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
      } else {
        error_prev_layer = error_begin;
      }
      
      
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron];
        weights = ann->weights + neuron_it->first_con;

        N=neuron_it->last_con - neuron_it->first_con;
        cudaMalloc(&d_errors,N*sizeof(fann_type));
      cudaMalloc(&d_wts,N*sizeof(fann_type));
        
      cudaMemcpy(d_errors,error_prev_layer,N*sizeof(fann_type),cudaMemcpyHostToDevice);
      cudaMemcpy(d_wts,weights,N*sizeof(fann_type),cudaMemcpyHostToDevice);
        

        numBlocks=(N+threadsperblock-1)/threadsperblock;
        
        backpropagate_gpu<<<numBlocks,threadsperblock>>>(N,d_errors,tmp_error,d_wts);

        cudaMemcpy(error_prev_layer,d_errors,N*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(weights,d_wts,N*sizeof(fann_type),cudaMemcpyDeviceToHost);

        cudaFree(d_errors);
        cudaFree(d_wts);
      }
      

    } else {
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron];
        weights = ann->weights + neuron_it->first_con;
        connections = ann->connections + neuron_it->first_con;
        for (i = neuron_it->last_con - neuron_it->first_con; i--;) {
          error_begin[connections[i] - first_neuron] += tmp_error * weights[i];
        }
      }
    }

    /* then calculate the actual errors in the previous layer */
    error_prev_layer = error_begin + ((layer_it - 1)->first_neuron - first_neuron);
    last_neuron = (layer_it - 1)->last_neuron;

    for (neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++) {
      *error_prev_layer *=
          fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
                                  neuron_it->value, neuron_it->sum);
      error_prev_layer++;
    }
  }
}

/* INTERNAL FUNCTION
   Update weights for incremental training
*/
// Device code
__global__ void Add_Mult_update_wts(fann_type* A, fann_type B, fann_type* C, const float constant, int N, fann_neuron* D)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
      fann_type delta_w = B * D[i].value + constant * A[i];
      C[i] += delta_w;
      A[i] = delta_w;
    }
}

void fann_update_weights(struct fann *ann) {
  
  struct fann_neuron *neuron_it, *last_neuron, *prev_neurons;
  fann_type tmp_error, *weights;
  struct fann_layer *layer_it;
  
  unsigned int num_connections;

  /* store some variabels local for fast access */
  const float learning_rate = ann->learning_rate;
  const float learning_momentum = ann->learning_momentum;
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  struct fann_layer *first_layer = ann->first_layer;
  const struct fann_layer *last_layer = ann->last_layer;
  fann_type *error_begin = ann->train_errors;
  fann_type *deltas_begin, *weights_deltas;

  // device parameters
  fann_type* d_wts;
  fann_type* d_wt_del;
  fann_neuron* d_prev_neurons;
  
  int threadsperblock=256;

  /* if no room allocated for the deltas, allocate it now */
  if (ann->prev_weights_deltas == NULL) {
    ann->prev_weights_deltas =
        (fann_type *)calloc(ann->total_connections_allocated, sizeof(fann_type));
    if (ann->prev_weights_deltas == NULL) {
      fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  }

  deltas_begin = ann->prev_weights_deltas;
  prev_neurons = first_neuron;
  for (layer_it = (first_layer + 1); layer_it != last_layer; layer_it++) {

    last_neuron = layer_it->last_neuron;
    if (ann->connection_rate >= 1) {
      if (ann->network_type == FANN_NETTYPE_LAYER) {
        prev_neurons = (layer_it - 1)->first_neuron;
      }
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron] * learning_rate;
        num_connections = neuron_it->last_con - neuron_it->first_con;
        weights = ann->weights + neuron_it->first_con;
        weights_deltas = deltas_begin + neuron_it->first_con;

        cudaMalloc(&d_wts,num_connections*sizeof(fann_type));
        cudaMalloc(&d_wt_del,num_connections*sizeof(fann_type));

        cudaMemcpy(d_wts,weights,num_connections*sizeof(fann_type),cudaMemcpyHostToDevice);
        cudaMemcpy(d_wt_del,weights_deltas,num_connections*sizeof(fann_type),cudaMemcpyHostToDevice);

        cudaMalloc(&d_prev_neurons,num_connections*sizeof(fann_neuron));
        cudaMemcpy(d_prev_neurons,prev_neurons,num_connections*sizeof(fann_neuron),cudaMemcpyHostToDevice);
        
        int numblocks=num_connections/threadsperblock;
        
        Add_Mult_update_wts<<<numblocks,threadsperblock>>>(d_wt_del,tmp_error,d_wts,learning_momentum,num_connections,d_prev_neurons);

        cudaMemcpy(weights,d_wts,num_connections*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(weights_deltas,d_wt_del,num_connections*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(prev_neurons,d_prev_neurons,num_connections*sizeof(fann_neuron),cudaMemcpyDeviceToHost);
        }
      }
     else {
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron] * learning_rate;
        num_connections = neuron_it->last_con - neuron_it->first_con;
        weights = ann->weights + neuron_it->first_con;
        weights_deltas = deltas_begin + neuron_it->first_con;
        
        cudaMalloc(&d_wts,num_connections*sizeof(fann_type));
        cudaMalloc(&d_wt_del,num_connections*sizeof(fann_type));

        cudaMemcpy(d_wts,weights,num_connections*sizeof(fann_type),cudaMemcpyHostToDevice);
        cudaMemcpy(d_wt_del,weights_deltas,num_connections*sizeof(fann_type),cudaMemcpyHostToDevice);

        cudaMalloc(&d_prev_neurons,num_connections*sizeof(fann_neuron));
        cudaMemcpy(d_prev_neurons,prev_neurons,num_connections*sizeof(fann_neuron),cudaMemcpyHostToDevice);
        
        
        int numblocks=num_connections/threadsperblock;
        Add_Mult_update_wts<<<numblocks,threadsperblock>>>(d_wt_del,tmp_error,d_wts,learning_momentum,num_connections,d_prev_neurons);
        
        cudaMemcpy(weights,d_wts,num_connections*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(weights_deltas,d_wt_del,num_connections*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(prev_neurons,d_prev_neurons,num_connections*sizeof(fann_neuron),cudaMemcpyDeviceToHost);

        cudaFree(d_wts);
        cudaFree(d_wt_del);
        cudaFree(d_prev_neurons);
      }
    }
  }
}



/* INTERNAL FUNCTION
   Update slopes for batch training
   layer_begin = ann->first_layer+1 and layer_end = ann->last_layer-1
   will update all slopes.
*/

__global__ void slopes_gpu(int N, fann_type* nrn_slope, fann_type tmp, fann_neuron* val)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
      nrn_slope[i]+=tmp*val[i].value;
    }

}

__global__ void slopes_gpu2(int N, fann_type* nrn_slope, fann_type tmp, fann_neuron** val)
{ 
  int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
      nrn_slope[i]+=tmp*val[i]->value;
    }

}

void fann_update_slopes_batch(struct fann *ann, struct fann_layer *layer_begin,
                              struct fann_layer *layer_end) {
  struct fann_neuron *neuron_it, *last_neuron, *prev_neurons, **connections;
  fann_type tmp_error;
  unsigned int num_connections;

  /* store some variabels local for fast access */
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  fann_type *error_begin = ann->train_errors;
  fann_type *slope_begin, *neuron_slope;

  /* if no room allocated for the slope variabels, allocate it now */
  if (ann->train_slopes == NULL) {
    ann->train_slopes = (fann_type *)calloc(ann->total_connections_allocated, sizeof(fann_type));
    if (ann->train_slopes == NULL) {
      fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  }

  if (layer_begin == NULL) {
    layer_begin = ann->first_layer + 1;
  }

  if (layer_end == NULL) {
    layer_end = ann->last_layer - 1;
  }

  slope_begin = ann->train_slopes;

  prev_neurons = first_neuron;

  // device parameters
  fann_type* d_nrn_slope;
  fann_neuron* d_prev_nrn;
  fann_neuron** d_connect;
  int threadsperblock=256;
  int numBlocks;

  for (; layer_begin <= layer_end; layer_begin++) {

    last_neuron = layer_begin->last_neuron;
    if (ann->connection_rate >= 1) {
      if (ann->network_type == FANN_NETTYPE_LAYER) {
        prev_neurons = (layer_begin - 1)->first_neuron;
      }

      for (neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron];
        neuron_slope = slope_begin + neuron_it->first_con;
        num_connections = neuron_it->last_con - neuron_it->first_con;

        cudaMalloc(&d_nrn_slope,num_connections*sizeof(fann_type));
        cudaMalloc(&d_prev_nrn,num_connections*sizeof(fann_neuron));

        cudaMemcpy(d_nrn_slope,neuron_slope,num_connections*sizeof(fann_type),cudaMemcpyHostToDevice);
        cudaMemcpy(d_prev_nrn,prev_neurons,num_connections*sizeof(fann_neuron),cudaMemcpyHostToDevice);
        
        numBlocks=(num_connections+threadsperblock-1)/threadsperblock;

        slopes_gpu<<<numBlocks,threadsperblock>>>(num_connections,d_nrn_slope,tmp_error,d_prev_nrn);

        cudaMemcpy(neuron_slope,d_nrn_slope,num_connections*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(prev_neurons,d_prev_nrn,num_connections*sizeof(fann_neuron),cudaMemcpyDeviceToHost);

        cudaFree(d_nrn_slope);
        cudaFree(d_prev_nrn);
      }
    } else {
      for (neuron_it = layer_begin->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_error = error_begin[neuron_it - first_neuron];
        neuron_slope = slope_begin + neuron_it->first_con;
        num_connections = neuron_it->last_con - neuron_it->first_con;
        connections = ann->connections + neuron_it->first_con;

        cudaMalloc(&d_nrn_slope,num_connections*sizeof(fann_type));
        cudaMalloc(&d_connect,num_connections*sizeof(fann_neuron));

        cudaMemcpy(d_nrn_slope,neuron_slope,num_connections*sizeof(fann_type),cudaMemcpyHostToDevice);
        cudaMemcpy(d_connect,connections,num_connections*sizeof(fann_neuron),cudaMemcpyHostToDevice);
        
        numBlocks=(num_connections+threadsperblock-1)/threadsperblock;

        slopes_gpu2<<<numBlocks,threadsperblock>>>(num_connections,d_nrn_slope,tmp_error,d_connect);

        cudaMemcpy(neuron_slope,d_nrn_slope,num_connections*sizeof(fann_type),cudaMemcpyDeviceToHost);
        cudaMemcpy(connections,d_connect,num_connections*sizeof(fann_neuron),cudaMemcpyDeviceToHost);

        cudaFree(d_nrn_slope);
        cudaFree(d_connect);

      }
    }
  }
}

/* INTERNAL FUNCTION
   Clears arrays used for training before a new training session.
   Also creates the arrays that do not exist yet.
 */
void fann_clear_train_arrays(struct fann *ann) {
  unsigned int i;
  fann_type delta_zero;

  /* if no room allocated for the slope variabels, allocate it now
   * (calloc clears mem) */
  if (ann->train_slopes == NULL) {
    ann->train_slopes = (fann_type *)calloc(ann->total_connections_allocated, sizeof(fann_type));
    if (ann->train_slopes == NULL) {
      fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  } else {
    memset(ann->train_slopes, 0, (ann->total_connections_allocated) * sizeof(fann_type));
  }

  /* if no room allocated for the variabels, allocate it now */
  if (ann->prev_steps == NULL) {
    ann->prev_steps = (fann_type *)malloc(ann->total_connections_allocated * sizeof(fann_type));
    if (ann->prev_steps == NULL) {
      fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  }

  if (ann->training_algorithm == FANN_TRAIN_RPROP) {
    delta_zero = ann->rprop_delta_zero;

    for (i = 0; i < ann->total_connections_allocated; i++) ann->prev_steps[i] = delta_zero;
  } else {
    memset(ann->prev_steps, 0, (ann->total_connections_allocated) * sizeof(fann_type));
  }

  /* if no room allocated for the variabels, allocate it now */
  if (ann->prev_train_slopes == NULL) {
    ann->prev_train_slopes =
        (fann_type *)calloc(ann->total_connections_allocated, sizeof(fann_type));
    if (ann->prev_train_slopes == NULL) {
      fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
      return;
    }
  } else {
    memset(ann->prev_train_slopes, 0, (ann->total_connections_allocated) * sizeof(fann_type));
  }
}

/* INTERNAL FUNCTION
   Update weights for batch training
 */
void fann_update_weights_batch(struct fann *ann, unsigned int num_data, unsigned int first_weight,
                               unsigned int past_end) {
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  const float epsilon = ann->learning_rate / num_data;
  unsigned int i = first_weight;

  for (; i != past_end; i++) {
    weights[i] += train_slopes[i] * epsilon;
    train_slopes[i] = 0.0;
  }
}

/* INTERNAL FUNCTION
   The quickprop training algorithm
 */

void fann_update_weights_quickprop(struct fann *ann, unsigned int num_data,
                                   unsigned int first_weight, unsigned int past_end) {
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;

  fann_type w, prev_step, slope, prev_slope, next_step;

  float epsilon = ann->learning_rate / num_data;
  float decay = ann->quickprop_decay; /*-0.0001;*/
  float mu = ann->quickprop_mu;       /*1.75; */
  float shrink_factor = (float)(mu / (1.0 + mu));

  unsigned int i = first_weight;

  for (; i != past_end; i++) {
    w = weights[i];
    prev_step = prev_steps[i];
    slope = train_slopes[i] + decay * w;
    prev_slope = prev_train_slopes[i];
    next_step = 0.0;

    /* The step must always be in direction opposite to the slope. */
    if (prev_step > 0.001) {
      /* If last step was positive...  */
      if (slope > 0.0) /*  Add in linear term if current slope is still positive. */
        next_step += epsilon * slope;

      /*If current slope is close to or larger than prev slope...  */
      if (slope > (shrink_factor * prev_slope))
        next_step += mu * prev_step; /* Take maximum size negative step. */
      else
        next_step += prev_step * slope / (prev_slope - slope); /* Else, use quadratic estimate. */
    } else if (prev_step < -0.001) {
      /* If last step was negative...  */
      if (slope < 0.0) /*  Add in linear term if current slope is still negative. */
        next_step += epsilon * slope;

      /* If current slope is close to or more neg than prev slope... */
      if (slope < (shrink_factor * prev_slope))
        next_step += mu * prev_step; /* Take maximum size negative step. */
      else
        next_step += prev_step * slope / (prev_slope - slope); /* Else, use quadratic estimate. */
    } else /* Last step was zero, so use only linear term. */
      next_step += epsilon * slope;

    /*
    if(next_step > 1000 || next_step < -1000)
    {
            printf("quickprop[%d] weight=%f, slope=%f, prev_slope=%f, next_step=%f, prev_step=%f\n",
                       i, weights[i], slope, prev_slope, next_step, prev_step);
               if(next_step > 1000)
               next_step = 1000;
               else
               next_step = -1000;
    }
*/

    /* update global data arrays */
    prev_steps[i] = next_step;

    w += next_step;

    if (w > 1500)
      weights[i] = 1500;
    else if (w < -1500)
      weights[i] = -1500;
    else
      weights[i] = w;

    /*weights[i] = w;*/

    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
  }
}

/* INTERNAL FUNCTION
   The iRprop- algorithm
*/
__global__ void gpu_update_wts(int begin, int end, fann_type* train_slopes, fann_type* prev_train_slopes, fann_type* prev_steps,
float increase_factor, float delta_max, float decrease_factor, float delta_min, fann_type* weights)
{
fann_type prev_step, slope, prev_slope, next_step, same_sign;
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i>=begin & i<end)
{
prev_step = fann_max(
        prev_steps[i],
        (fann_type)0.0001); /* prev_step may not be zero because then the training will stop */
    slope = train_slopes[i];
    prev_slope = prev_train_slopes[i];

    same_sign = prev_slope * slope;

    if (same_sign >= 0.0)
      next_step = fann_min(prev_step * increase_factor, delta_max);
    else {
      next_step = fann_max(prev_step * decrease_factor, delta_min);
      slope = 0;
    }

    if (slope < 0) {
      weights[i] -= next_step;
      if (weights[i] < -1500) weights[i] = -1500;
    } else {
      weights[i] += next_step;
      if (weights[i] > 1500) weights[i] = 1500;
    }

    /*if(i == 2){
     * printf("weight=%f, slope=%f, next_step=%f, prev_step=%f\n", weights[i], slope, next_step,
     * prev_step);
     * } */

    /* update global data arrays */
    prev_steps[i] = next_step;
    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
}
}

void fann_update_weights_irpropm(struct fann *ann, unsigned int first_weight,
                                 unsigned int past_end) {
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;

  float increase_factor = ann->rprop_increase_factor; /*1.2; */
  float decrease_factor = ann->rprop_decrease_factor; /*0.5; */
  float delta_min = ann->rprop_delta_min;             /*0.0; */
  float delta_max = ann->rprop_delta_max;             /*50.0; */

  unsigned int i = first_weight;

  // device parameters
  fann_type* d_train_slopes;
  fann_type* d_prev_train_slopes; 
  fann_type* d_prev_steps;
  fann_type* d_weights;
  
  int N=past_end-first_weight;

  cudaMalloc(&d_train_slopes,N*sizeof(fann_type));
  cudaMalloc(&d_prev_train_slopes,N*sizeof(fann_type));
  cudaMalloc(&d_prev_steps,N*sizeof(fann_type));
  cudaMalloc(&d_weights,N*sizeof(fann_type));

  cudaMemcpy(d_train_slopes,train_slopes,N*sizeof(fann_type),cudaMemcpyHostToDevice);
  cudaMemcpy(d_prev_train_slopes,prev_train_slopes,N*sizeof(fann_type),cudaMemcpyHostToDevice);
  cudaMemcpy(d_prev_steps,prev_steps,N*sizeof(fann_type),cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights,weights,N*sizeof(fann_type),cudaMemcpyHostToDevice);

  int threadsperblock=256;
  int numBlocks=(N+threadsperblock-1)/threadsperblock;

  gpu_update_wts<<<numBlocks,threadsperblock>>>(i,past_end,d_train_slopes,d_prev_train_slopes,d_prev_steps,increase_factor,delta_max,decrease_factor,delta_min,d_weights);
  
  cudaMemcpy(train_slopes,d_train_slopes,N*sizeof(fann_type),cudaMemcpyDeviceToHost);
  cudaMemcpy(prev_train_slopes,d_prev_train_slopes,N*sizeof(fann_type),cudaMemcpyDeviceToHost);
  cudaMemcpy(prev_steps,d_prev_steps,N*sizeof(fann_type),cudaMemcpyDeviceToHost);
  cudaMemcpy(weights,d_weights,N*sizeof(fann_type),cudaMemcpyDeviceToHost);

  cudaFree(d_train_slopes);
  cudaFree(d_prev_steps);
  cudaFree(d_prev_train_slopes);
  cudaFree(d_weights);

}

/* INTERNAL FUNCTION
   The SARprop- algorithm
*/
void fann_update_weights_sarprop(struct fann *ann, unsigned int epoch, unsigned int first_weight,
                                 unsigned int past_end) {
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;

  fann_type prev_step, slope, prev_slope, next_step = 0, same_sign;

  /* These should be set from variables */
  float increase_factor = ann->rprop_increase_factor; /*1.2; */
  float decrease_factor = ann->rprop_decrease_factor; /*0.5; */
  /* TODO: why is delta_min 0.0 in iRprop? SARPROP uses 1x10^-6 (Braun and Riedmiller, 1993) */
  float delta_min = 0.000001f;
  float delta_max = ann->rprop_delta_max;                     /*50.0; */
  float weight_decay_shift = ann->sarprop_weight_decay_shift; /* ld 0.01 = -6.644 */
  float step_error_threshold_factor = ann->sarprop_step_error_threshold_factor; /* 0.1 */
  float step_error_shift = ann->sarprop_step_error_shift;                       /* ld 3 = 1.585 */
  float T = ann->sarprop_temperature;
  float MSE = fann_get_MSE(ann);
  float RMSE = sqrtf(MSE);

  unsigned int i = first_weight;

  /* for all weights; TODO: are biases included? */
  for (; i != past_end; i++) {
    /* TODO: confirm whether 1x10^-6 == delta_min is really better */
    prev_step = fann_max(
        prev_steps[i],
        (fann_type)0.000001); /* prev_step may not be zero because then the training will stop */
    /* calculate SARPROP slope; TODO: better as new error function? (see SARPROP paper)*/
    slope = -train_slopes[i] - weights[i] * (fann_type)fann_exp2(-T * epoch + weight_decay_shift);

    /* TODO: is prev_train_slopes[i] 0.0 in the beginning? */
    prev_slope = prev_train_slopes[i];

    same_sign = prev_slope * slope;

    if (same_sign > 0.0) {
      next_step = fann_min(prev_step * increase_factor, delta_max);
      /* TODO: are the signs inverted? see differences between SARPROP paper and iRprop */
      if (slope < 0.0)
        weights[i] += next_step;
      else
        weights[i] -= next_step;
    } else if (same_sign < 0.0) {
      if (prev_step < step_error_threshold_factor * MSE)
        next_step =
            prev_step * decrease_factor +
            (float)rand() / RAND_MAX * RMSE * (fann_type)fann_exp2(-T * epoch + step_error_shift);
      else
        next_step = fann_max(prev_step * decrease_factor, delta_min);

      slope = 0.0;
    } else {
      if (slope < 0.0)
        weights[i] += prev_step;
      else
        weights[i] -= prev_step;
    }

    /*if(i == 2){
     * printf("weight=%f, slope=%f, next_step=%f, prev_step=%f\n", weights[i], slope, next_step,
     * prev_step);
     * } */

    /* update global data arrays */
    prev_steps[i] = next_step;
    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
  }
}

   struct fann *FANN_API fann_create_standard_array(unsigned int num_layers,
                                                               const unsigned int *layers);

/* Function: fann_create_sparse
        Creates a standard backpropagation neural network, which is not fully connected.
        Parameters:
                connection_rate - The connection rate controls how many connections there will be in
   the network. If the connection rate is set to 1, the network will be fully connected, but if it
   is set to 0.5 only half of the connections will be set. A connection rate of 1 will yield the
   same result as <fann_create_standard> num_layers - The total number of layers including the input
   and the output layer.
                ... - Integer values determining the number of neurons in each layer starting with
   the input layer and ending with the output layer.
        Returns:
                A pointer to the newly created <struct fann>.
        See also:
                <fann_create_sparse_array>, <fann_create_standard>, <fann_create_shortcut>
        This function appears in FANN >= 2.0.0.
*/
   struct fann *FANN_API fann_create_sparse(float connection_rate,
                                                       unsigned int num_layers, ...);

/* Function: fann_create_sparse_array
   Just like <fann_create_sparse>, but with an array of layer sizes
   instead of individual parameters.
        See <fann_create_standard_array> for a description of the parameters.
        See also:
                <fann_create_sparse>, <fann_create_standard>, <fann_create_shortcut>
        This function appears in FANN >= 2.0.0.
*/
   struct fann *FANN_API fann_create_sparse_array(float connection_rate,
                                                             unsigned int num_layers,
                                                             const unsigned int *layers);

/* Function: fann_create_shortcut
        Creates a standard backpropagation neural network, which is fully connected and which
        also has shortcut connections.
        Shortcut connections are connections that skip layers. A fully connected network with
   shortcut connections is a network where all neurons are connected to all neurons in later layers.
        Including direct connections from the input layer to the output layer.
        See <fann_create_standard> for a description of the parameters.
        See also:
                <fann_create_shortcut_array>, <fann_create_standard>, <fann_create_sparse>,
        This function appears in FANN >= 2.0.0.
*/
   struct fann *FANN_API fann_create_shortcut(unsigned int num_layers, ...);

/* Function: fann_create_shortcut_array
   Just like <fann_create_shortcut>, but with an array of layer sizes
   instead of individual parameters.
        See <fann_create_standard_array> for a description of the parameters.
        See also:
                <fann_create_shortcut>, <fann_create_standard>, <fann_create_sparse>
        This function appears in FANN >= 2.0.0.
*/
   struct fann *FANN_API fann_create_shortcut_array(unsigned int num_layers,
                                                               const unsigned int *layers);
/* Function: fann_destroy
   Destroys the entire network, properly freeing all the associated memory.
        This function appears in FANN >= 1.0.0.
*/
   void FANN_API fann_destroy(struct fann *ann);

/* Function: fann_copy
   Creates a copy of a fann structure.
   Data in the user data <fann_set_user_data> is not copied, but the user data pointer is copied.
        This function appears in FANN >= 2.2.0.
*/
   struct fann *FANN_API fann_copy(struct fann *ann);

/* Function: fann_run
        Will run input through the neural network, returning an array of outputs, the number of
   which being equal to the number of neurons in the output layer.
        See also:
                <fann_test>
        This function appears in FANN >= 1.0.0.
*/
   fann_type *FANN_API fann_run(struct fann *ann, fann_type *input);

/* Function: fann_randomize_weights
        Give each connection a random weight between *min_weight* and *max_weight*
        From the beginning the weights are random between -0.1 and 0.1.
        See also:
                <fann_init_weights>
        This function appears in FANN >= 1.0.0.
*/
   void FANN_API fann_randomize_weights(struct fann *ann, fann_type min_weight,
                                                   fann_type max_weight);

/* Function: fann_init_weights
        Initialize the weights using Widrow + Nguyen's algorithm.
        This function behaves similarly to fann_randomize_weights. It will use the algorithm
   developed by Derrick Nguyen and Bernard Widrow to set the weights in such a way as to speed up
   training. This technique is not always successful, and in some cases can be less efficient than a
   purely random initialization.
        The algorithm requires access to the range of the input data (ie, largest and smallest
   input), and therefore accepts a second argument, data, which is the training data that will be
   used to train the network.
        See also:
                <fann_randomize_weights>, <fann_read_train_from_file>
        This function appears in FANN >= 1.1.0.
*/
   void FANN_API fann_init_weights(struct fann *ann, struct fann_train_data *train_data);

/* Function: fann_print_connections
        Will print the connections of the ann in a compact matrix, for easy viewing of the internals
        of the ann.
        The output from fann_print_connections on a small (2 2 1) network trained on the xor problem
        >Layer / Neuron 012345
        >L   1 / N    3 BBa...
        >L   1 / N    4 BBA...
        >L   1 / N    5 ......
        >L   2 / N    6 ...BBA
        >L   2 / N    7 ......
        This network has five real neurons and two bias neurons. This gives a total of seven neurons
        named from 0 to 6. The connections between these neurons can be seen in the matrix. "." is a
        place where there is no connection, while a character tells how strong the connection is on
   a scale from a-z. The two real neurons in the hidden layer (neuron 3 and 4 in layer 1) have
        connections from the three neurons in the previous layer as is visible in the first two
   lines. The output neuron (6) has connections from the three neurons in the hidden layer 3 - 5 as
   is visible in the fourth line.
        To simplify the matrix output neurons are not visible as neurons that connections can come
   from, and input and bias neurons are not visible as neurons that connections can go to.
        This function appears in FANN >= 1.2.0.
*/
   void FANN_API fann_print_connections(struct fann *ann);

/* Group: Parameters */
/* Function: fann_print_parameters
        Prints all of the parameters and options of the ANN
        This function appears in FANN >= 1.2.0.
*/
   void FANN_API fann_print_parameters(struct fann *ann);

/* Function: fann_get_num_input
   Get the number of input neurons.
        This function appears in FANN >= 1.0.0.
*/
   unsigned int FANN_API fann_get_num_input(struct fann *ann);

/* Function: fann_get_num_output
   Get the number of output neurons.
        This function appears in FANN >= 1.0.0.
*/
   unsigned int FANN_API fann_get_num_output(struct fann *ann);

/* Function: fann_get_total_neurons
   Get the total number of neurons in the entire network. This number does also include the
        bias neurons, so a 2-4-2 network has 2+4+2 +2(bias) = 10 neurons.
        This function appears in FANN >= 1.0.0.
*/
   unsigned int FANN_API fann_get_total_neurons(struct fann *ann);

/* Function: fann_get_total_connections
   Get the total number of connections in the entire network.
        This function appears in FANN >= 1.0.0.
*/
   unsigned int FANN_API fann_get_total_connections(struct fann *ann);

/* Function: fann_get_network_type
    Get the type of neural network it was created as.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
        Returns:
        The neural network type from enum <fann_network_type_enum>
    See Also:
        <fann_network_type_enum>
   This function appears in FANN >= 2.1.0
*/
   enum fann_nettype_enum FANN_API fann_get_network_type(struct fann *ann);

/* Function: fann_get_connection_rate
    Get the connection rate used when the network was created
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
        Returns:
        The connection rate
   This function appears in FANN >= 2.1.0
*/
   float FANN_API fann_get_connection_rate(struct fann *ann);

/* Function: fann_get_num_layers
    Get the number of layers in the network
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
        Returns:
                The number of layers in the neural network
        Example:
                > // Obtain the number of layers in a neural network
                > struct fann *ann = fann_create_standard(4, 2, 8, 9, 1);
        > unsigned int num_layers = fann_get_num_layers(ann);
   This function appears in FANN >= 2.1.0
*/
   unsigned int FANN_API fann_get_num_layers(struct fann *ann);

/*Function: fann_get_layer_array
    Get the number of neurons in each layer in the network.
    Bias is not included so the layers match the fann_create functions.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
    The layers array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long.
   This function appears in FANN >= 2.1.0
*/
   void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers);

/* Function: fann_get_bias_array
    Get the number of bias in each layer in the network.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
    The bias array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long.
   This function appears in FANN >= 2.1.0
*/
   void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias);

/* Function: fann_get_connection_array
    Get the connections in the network.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
    The connections array must be preallocated to at least
    sizeof(struct fann_connection) * fann_get_total_connections() long.
   This function appears in FANN >= 2.1.0
*/
   void FANN_API fann_get_connection_array(struct fann *ann,
                                                      struct fann_connection *connections);

/* Function: fann_set_weight_array
    Set connections in the network.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
    Only the weights can be changed, connections and weights are ignored
    if they do not already exist in the network.
    The array must have sizeof(struct fann_connection) * num_connections size.
   This function appears in FANN >= 2.1.0
*/
   void FANN_API fann_set_weight_array(struct fann *ann,
                                                  struct fann_connection *connections,
                                                  unsigned int num_connections);

/* Function: fann_set_weight
    Set a connection in the network.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
    Only the weights can be changed. The connection/weight is
    ignored if it does not already exist in the network.
   This function appears in FANN >= 2.1.0
*/
   void FANN_API fann_set_weight(struct fann *ann, unsigned int from_neuron,
                                            unsigned int to_neuron, fann_type weight);

/* Function: fann_get_weights
    Get all the network weights.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
                weights - A fann_type pointer to user data. It is the responsibility
                        of the user to allocate sufficient space to store all the weights.
   This function appears in FANN >= x.y.z
*/
   void FANN_API fann_get_weights(struct fann *ann, fann_type *weights);

/* Function: fann_set_weights
    Set network weights.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
                weights - A fann_type pointer to user data. It is the responsibility
                        of the user to make the weights array sufficient long
                        to store all the weights.
   This function appears in FANN >= x.y.z
*/
   void FANN_API fann_set_weights(struct fann *ann, fann_type *weights);

/* Function: fann_set_user_data
    Store a pointer to user defined data. The pointer can be
    retrieved with <fann_get_user_data> for example in a
    callback. It is the user's responsibility to allocate and
    deallocate any data that the pointer might point to.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
                user_data - A void pointer to user defined data.
   This function appears in FANN >= 2.1.0
*/
   void FANN_API fann_set_user_data(struct fann *ann, void *user_data);

/* Function: fann_get_user_data
    Get a pointer to user defined data that was previously set
    with <fann_set_user_data>. It is the user's responsibility to
    allocate and deallocate any data that the pointer might point to.
    Parameters:
                ann - A previously created neural network structure of
            type <struct fann> pointer.
    Returns:
        A void pointer to user defined data.
   This function appears in FANN >= 2.1.0
*/
   void *FANN_API fann_get_user_data(struct fann *ann);

/* Function: fann_disable_seed_rand
   Disables the automatic random generator seeding that happens in FANN.
   Per default FANN will always seed the random generator when creating a new network,
   unless FANN_NO_SEED is defined during compilation of the library. This method can
   disable this at runtime.
   This function appears in FANN >= 2.3.0
*/
   void FANN_API fann_disable_seed_rand();

/* Function: fann_enable_seed_rand
   Enables the automatic random generator seeding that happens in FANN.
   Per default FANN will always seed the random generator when creating a new network,
   unless FANN_NO_SEED is defined during compilation of the library. This method can
   disable this at runtime.
   This function appears in FANN >= 2.3.0
*/
   void FANN_API fann_enable_seed_rand();

unsigned int FANN_API fann_get_num_output(struct fann *ann)
{
  return ann->num_output;
}

struct fann *FANN_API fann_create_standard(unsigned int num_layers, ...) {
  struct fann *ann;
  va_list layer_sizes;
  int i;
  int status;
  int arg;
  unsigned int *layers = (unsigned int *)calloc(num_layers, sizeof(unsigned int));

  if (layers == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  va_start(layer_sizes, num_layers);

  status = 1;
  for (i = 0; i < (int)num_layers; i++) {
    arg = va_arg(layer_sizes, unsigned int);
    if (arg < 0 || arg > 1000000) status = 0;
    layers[i] = arg;
  }
  va_end(layer_sizes);

  if (!status) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    free(layers);
    return NULL;
  }

  ann = fann_create_standard_array(num_layers, layers);

  free(layers);

  return ann;
}

 struct fann *FANN_API fann_create_standard_array(unsigned int num_layers,
                                                               const unsigned int *layers) {
  return fann_create_sparse_array(1, num_layers, layers);
}

  struct fann *FANN_API fann_create_sparse(float connection_rate,
                                                       unsigned int num_layers, ...) {
  struct fann *ann;
  va_list layer_sizes;
  int i;
  int status;
  int arg;
  unsigned int *layers = (unsigned int *)calloc(num_layers, sizeof(unsigned int));

  va_start(layer_sizes, num_layers);
  status = 1;
  for (i = 0; i < (int)num_layers; i++) {
    arg = va_arg(layer_sizes, unsigned int);
    if (arg < 0 || arg > 1000000) status = 0;
    layers[i] = arg;
  }
  va_end(layer_sizes);

  if (!status) {
    
    free(layers);
    return NULL;
  }

  ann = fann_create_sparse_array(connection_rate, num_layers, layers);
  free(layers);

  return ann;
}

  struct fann *FANN_API fann_create_sparse_array(float connection_rate,
                                                             unsigned int num_layers,
                                                             const unsigned int *layers) {
  struct fann_layer *layer_it, *last_layer, *prev_layer;
  struct fann *ann;
  struct fann_neuron *neuron_it, *last_neuron, *random_neuron, *bias_neuron;

  unsigned int num_neurons_in, num_neurons_out, i, j;
  unsigned int min_connections, max_connections, num_connections;
  unsigned int connections_per_neuron, allocated_connections;
  unsigned int random_number, found_connection, tmp_con;

  if (connection_rate > 1) {
    connection_rate = 1;
  }

  fann_seed_rand();

  /* allocate the general structure */
  ann = fann_allocate_structure(num_layers);
  if (ann == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  ann->connection_rate = connection_rate;

  /* determine how many neurons there should be in each layer */
  i = 0;
  for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
    /* we do not allocate room here, but we make sure that
     * last_neuron - first_neuron is the number of neurons */
    layer_it->first_neuron = NULL;
    layer_it->last_neuron = layer_it->first_neuron + layers[i++] + 1; /* +1 for bias */
    ann->total_neurons += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
  }

  ann->num_output =
      (unsigned int)((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron - 1);
  ann->num_input =
      (unsigned int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);

  /* allocate room for the actual neurons */
  fann_allocate_neurons(ann);
  if (ann->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
    fann_destroy(ann);
    return NULL;
  }


  num_neurons_in = ann->num_input;
  for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
    num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron - 1);
    /*if all neurons in each layer should be connected to at least one neuron
     * in the previous layer, and one neuron in the next layer.
     * and the bias node should be connected to the all neurons in the next layer.
     * Then this is the minimum amount of neurons */
    min_connections = fann_max(num_neurons_in, num_neurons_out); /* not calculating bias */
    max_connections = num_neurons_in * num_neurons_out;          /* not calculating bias */
    num_connections =
        fann_max(min_connections, (unsigned int)(0.5 + (connection_rate * max_connections))) +
        num_neurons_out;

    connections_per_neuron = num_connections / num_neurons_out;
    allocated_connections = 0;
    /* Now split out the connections on the different neurons */
    for (i = 0; i != num_neurons_out; i++) {
      layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
      allocated_connections += connections_per_neuron;
      layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;

      layer_it->first_neuron[i].activation_function = FANN_SIGMOID_STEPWISE;

      layer_it->first_neuron[i].activation_steepness = 0.5;


      if (allocated_connections < (num_connections * (i + 1)) / num_neurons_out) {
        layer_it->first_neuron[i].last_con++;
        allocated_connections++;
      }
    }

    /* bias neuron also gets stuff */
    layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
    layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;

    ann->total_connections += num_connections;

    /* used in the next run of the loop */
    num_neurons_in = num_neurons_out;
  }

  fann_allocate_connections(ann);
  if (ann->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
    fann_destroy(ann);
    return NULL;
  }

  if (connection_rate >= 1) {

    prev_layer = ann->first_layer;
    last_layer = ann->last_layer;
    for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
      last_neuron = layer_it->last_neuron - 1;
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        tmp_con = neuron_it->last_con - 1;
        for (i = neuron_it->first_con; i != tmp_con; i++) {
          ann->weights[i] = (fann_type)fann_random_weight();
          /* these connections are still initialized for fully connected networks, to allow
           * operations to work, that are not optimized for fully connected networks.
           */
          ann->connections[i] = prev_layer->first_neuron + (i - neuron_it->first_con);
        }

        /* bias weight */
        ann->weights[tmp_con] = (fann_type)fann_random_bias_weight();
        ann->connections[tmp_con] = prev_layer->first_neuron + (tmp_con - neuron_it->first_con);
      }

    }
  } else {
    /* make connections for a network, that are not fully connected */

    /* generally, what we do is first to connect all the input
     * neurons to a output neuron, respecting the number of
     * available input neurons for each output neuron. Then
     * we go through all the output neurons, and connect the
     * rest of the connections to input neurons, that they are
     * not allready connected to.
     */

    /* All the connections are cleared by calloc, because we want to
     * be able to see which connections are allready connected */

    for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
      num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron - 1);
      num_neurons_in =
          (unsigned int)((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron - 1);

      /* first connect the bias neuron */
      bias_neuron = (layer_it - 1)->last_neuron - 1;
      last_neuron = layer_it->last_neuron - 1;
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        ann->connections[neuron_it->first_con] = bias_neuron;
        ann->weights[neuron_it->first_con] = (fann_type)fann_random_bias_weight();
      }

      /* then connect all neurons in the input layer */
      last_neuron = (layer_it - 1)->last_neuron - 1;
      for (neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++) {
        /* random neuron in the output layer that has space
         * for more connections */
        do {
          random_number = (int)(0.5 + fann_rand(0, num_neurons_out - 1));
          random_neuron = layer_it->first_neuron + random_number;
          /* checks the last space in the connections array for room */
        } while (ann->connections[random_neuron->last_con - 1]);

        /* find an empty space in the connection array and connect */
        for (i = random_neuron->first_con; i < random_neuron->last_con; i++) {
          if (ann->connections[i] == NULL) {
            ann->connections[i] = neuron_it;
            ann->weights[i] = (fann_type)fann_random_weight();
            break;
          }
        }
      }

      /* then connect the rest of the unconnected neurons */
      last_neuron = layer_it->last_neuron - 1;
      for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
        /* find empty space in the connection array and connect */
        for (i = neuron_it->first_con; i < neuron_it->last_con; i++) {
          /* continue if allready connected */
          if (ann->connections[i] != NULL) continue;

          do {
            found_connection = 0;
            random_number = (int)(0.5 + fann_rand(0, num_neurons_in - 1));
            random_neuron = (layer_it - 1)->first_neuron + random_number;

            /* check to see if this connection is allready there */
            for (j = neuron_it->first_con; j < i; j++) {
              if (random_neuron == ann->connections[j]) {
                found_connection = 1;
                break;
              }
            }

          } while (found_connection);

          /* we have found a neuron that is not allready
           * connected to us, connect it */
          ann->connections[i] = random_neuron;
          ann->weights[i] = (fann_type)fann_random_weight();
        }
      }


    }

    /* TODO it would be nice to have the randomly created
     * connections sorted for smoother memory access.
     */
  }



  return ann;
}

  struct fann *FANN_API fann_create_shortcut(unsigned int num_layers, ...) {
  struct fann *ann;
  int i;
  int status;
  int arg;
  va_list layer_sizes;
  unsigned int *layers = (unsigned int *)calloc(num_layers, sizeof(unsigned int));

  if (layers == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  va_start(layer_sizes, num_layers);
  status = 1;
  for (i = 0; i < (int)num_layers; i++) {
    arg = va_arg(layer_sizes, unsigned int);
    if (arg < 0 || arg > 1000000) status = 0;
    layers[i] = arg;
  }
  va_end(layer_sizes);

  if (!status) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    free(layers);
    return NULL;
  }

  ann = fann_create_shortcut_array(num_layers, layers);

  free(layers);

  return ann;
}

  struct fann *FANN_API fann_create_shortcut_array(unsigned int num_layers,
                                                               const unsigned int *layers) {
  struct fann_layer *layer_it, *layer_it2, *last_layer;
  struct fann *ann;
  struct fann_neuron *neuron_it, *neuron_it2 = 0;
  unsigned int i;
  unsigned int num_neurons_in, num_neurons_out;

  fann_seed_rand();

  /* allocate the general structure */
  ann = fann_allocate_structure(num_layers);
  if (ann == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  ann->connection_rate = 1;
  ann->network_type = FANN_NETTYPE_SHORTCUT;

  /* determine how many neurons there should be in each layer */
  i = 0;
  for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
    /* we do not allocate room here, but we make sure that
     * last_neuron - first_neuron is the number of neurons */
    layer_it->first_neuron = NULL;
    layer_it->last_neuron = layer_it->first_neuron + layers[i++];
    if (layer_it == ann->first_layer) {
      /* there is a bias neuron in the first layer */
      layer_it->last_neuron++;
    }

    ann->total_neurons += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
  }

  ann->num_output =
      (unsigned int)((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron);
  ann->num_input =
      (unsigned int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);

  /* allocate room for the actual neurons */
  fann_allocate_neurons(ann);
  if (ann->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
    fann_destroy(ann);
    return NULL;
  }



  num_neurons_in = ann->num_input;
  last_layer = ann->last_layer;
  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
    num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);

    /* Now split out the connections on the different neurons */
    for (i = 0; i != num_neurons_out; i++) {
      layer_it->first_neuron[i].first_con = ann->total_connections;
      ann->total_connections += num_neurons_in + 1;
      layer_it->first_neuron[i].last_con = ann->total_connections;

      layer_it->first_neuron[i].activation_function = FANN_SIGMOID_STEPWISE;

      layer_it->first_neuron[i].activation_steepness = 0.5;

    }


    /* used in the next run of the loop */
    num_neurons_in += num_neurons_out;
  }

  fann_allocate_connections(ann);
  if (ann->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
    fann_destroy(ann);
    return NULL;
  }

  /* Connections are created from all neurons to all neurons in later layers
   */
  num_neurons_in = ann->num_input + 1;
  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
    for (neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++) {
      i = neuron_it->first_con;
      for (layer_it2 = ann->first_layer; layer_it2 != layer_it; layer_it2++) {
        for (neuron_it2 = layer_it2->first_neuron; neuron_it2 != layer_it2->last_neuron;
             neuron_it2++) {
          ann->weights[i] = (fann_type)fann_random_weight();
          ann->connections[i] = neuron_it2;
          i++;
        }
      }
    }
    num_neurons_in += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
  }



  return ann;
}

__global__ void assign_fann_run(fann_type* out,fann_neuron* nrn, int N)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
      out[i]=nrn[i].value;
    }

}

__global__ void assign2_fann_run(fann_neuron* out,fann_type* nrn, int N)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
      out[i].value=nrn[i];
    }

}

  fann_type *FANN_API fann_run(struct fann *ann, fann_type *input) {
  struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
  unsigned int i, num_connections, num_input, num_output;
  fann_type neuron_sum, *output;
  fann_type *weights;
  struct fann_layer *layer_it, *last_layer;
  unsigned int activation_function;
  fann_type steepness;
  
  int threadsperblock, numBlocks;
  /* store some variabels local for fast access */
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

  fann_type max_sum = 0;
  
  //device parameters
  fann_neuron* d_first_neuron;
  fann_type* d_input;


  /* first set the input */
  num_input = ann->num_input;

  cudaMalloc(&d_first_neuron,num_input*sizeof(fann_neuron));
  cudaMalloc(&d_input,num_input*sizeof(fann_type));

  cudaMemcpy(d_first_neuron,first_neuron,num_input*sizeof(fann_neuron),cudaMemcpyHostToDevice);
  cudaMemcpy(d_input,input,num_input*sizeof(fann_type),cudaMemcpyHostToDevice);
  
  threadsperblock=256;
  numBlocks = (num_input + threadsperblock - 1) / threadsperblock;
  dim3 dimBlock(threadsperblock);
  dim3 dimGrid(numBlocks);

  assign2_fann_run<<<dimGrid,dimBlock>>>(d_first_neuron,d_input,num_input);
  
  //cudaDeviceSynchronize();

  cudaMemcpy(first_neuron,d_first_neuron,num_input*sizeof(fann_neuron),cudaMemcpyDeviceToHost);
  cudaMemcpy(input,d_input,num_input*sizeof(fann_type),cudaMemcpyDeviceToHost);
  
  cudaFree(d_first_neuron);
  cudaFree(d_input);

  /* Set the bias neuron in the input layer */

  (ann->first_layer->last_neuron - 1)->value = 1;


  last_layer = ann->last_layer;
  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
    last_neuron = layer_it->last_neuron;
    for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
      if (neuron_it->first_con == neuron_it->last_con) {
        /* bias neurons */

        neuron_it->value = 1;

        continue;
      }

      activation_function = neuron_it->activation_function;
      steepness = neuron_it->activation_steepness;

      neuron_sum = 0;
      num_connections = neuron_it->last_con - neuron_it->first_con;
      weights = ann->weights + neuron_it->first_con;

      if (ann->connection_rate >= 1) {
        if (ann->network_type == FANN_NETTYPE_SHORTCUT) {
          neurons = ann->first_layer->first_neuron;
        } else {
          neurons = (layer_it - 1)->first_neuron;
        }

        /* unrolled loop start */
        i = num_connections & 3; /* same as modulo 4 */
        switch (i) {
          case 3:
            neuron_sum += fann_mult(weights[2], neurons[2].value);
          case 2:
            neuron_sum += fann_mult(weights[1], neurons[1].value);
          case 1:
            neuron_sum += fann_mult(weights[0], neurons[0].value);
          case 0:
            break;
        }

        for (; i != num_connections; i += 4) {
          neuron_sum += fann_mult(weights[i], neurons[i].value) +
                        fann_mult(weights[i + 1], neurons[i + 1].value) +
                        fann_mult(weights[i + 2], neurons[i + 2].value) +
                        fann_mult(weights[i + 3], neurons[i + 3].value);
        }
        /* unrolled loop end */

        /*
         * for(i = 0;i != num_connections; i++){
         * printf("%f += %f*%f, ", neuron_sum, weights[i], neurons[i].value);
         * neuron_sum += fann_mult(weights[i], neurons[i].value);
         * }
         */
      } else {
        neuron_pointers = ann->connections + neuron_it->first_con;

        i = num_connections & 3; /* same as modulo 4 */
        switch (i) {
          case 3:
            neuron_sum += fann_mult(weights[2], neuron_pointers[2]->value);
          case 2:
            neuron_sum += fann_mult(weights[1], neuron_pointers[1]->value);
          case 1:
            neuron_sum += fann_mult(weights[0], neuron_pointers[0]->value);
          case 0:
            break;
        }

        for (; i != num_connections; i += 4) {
          neuron_sum += fann_mult(weights[i], neuron_pointers[i]->value) +
                        fann_mult(weights[i + 1], neuron_pointers[i + 1]->value) +
                        fann_mult(weights[i + 2], neuron_pointers[i + 2]->value) +
                        fann_mult(weights[i + 3], neuron_pointers[i + 3]->value);
        }
      }


      neuron_sum = fann_mult(steepness, neuron_sum);

      max_sum = 150 / steepness;
      if (neuron_sum > max_sum)
        neuron_sum = max_sum;
      else if (neuron_sum < -max_sum)
        neuron_sum = -max_sum;

      neuron_it->sum = neuron_sum;

      fann_activation_switch(activation_function, neuron_sum, neuron_it->value);

    }
  }

  /* set the output */
  output = ann->output;
  num_output = ann->num_output;
  neurons = (ann->last_layer - 1)->first_neuron;

  // device parameters
  fann_type* d_output;
  fann_neuron* d_neuron;

  cudaMalloc(&d_output,num_output*sizeof(fann_type));
  cudaMalloc(&d_neuron,num_output*sizeof(fann_neuron));

  cudaMemcpy(d_output,output,num_output*sizeof(fann_type),cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron,neurons,num_output*sizeof(fann_neuron),cudaMemcpyHostToDevice);
  
  threadsperblock=256;
  numBlocks = (num_output + threadsperblock - 1) / threadsperblock;
  dim3 dimBlock2(threadsperblock);
  dim3 dimGrid2(numBlocks);

  assign_fann_run<<<dimGrid2,dimBlock2>>>(d_output,d_neuron,num_output);
  
  //cudaDeviceSynchronize();

  cudaMemcpy(output,d_output,num_output*sizeof(fann_type),cudaMemcpyDeviceToHost);
  cudaMemcpy(neurons,d_neuron,num_output*sizeof(fann_neuron),cudaMemcpyDeviceToHost);
  
  cudaFree(d_output);
  cudaFree(d_neuron);
 
  return ann->output;

}



  void FANN_API fann_destroy(struct fann *ann) {
  if (ann == NULL) return;
  fann_safe_free(ann->weights);
  fann_safe_free(ann->connections);
  fann_safe_free(ann->first_layer->first_neuron);
  fann_safe_free(ann->first_layer);
  fann_safe_free(ann->output);
  fann_safe_free(ann->train_errors);
  fann_safe_free(ann->train_slopes);
  fann_safe_free(ann->prev_train_slopes);
  fann_safe_free(ann->prev_steps);
  fann_safe_free(ann->prev_weights_deltas);
  fann_safe_free(ann->errstr);
  fann_safe_free(ann->cascade_activation_functions);
  fann_safe_free(ann->cascade_activation_steepnesses);
  fann_safe_free(ann->cascade_candidate_scores);



  fann_safe_free(ann);
}

  void FANN_API fann_randomize_weights(struct fann *ann, fann_type min_weight,
                                                   fann_type max_weight) {
  fann_type *last_weight;
  fann_type *weights = ann->weights;

  last_weight = weights + ann->total_connections;
  for (; weights != last_weight; weights++) {
    *weights = (fann_type)(fann_rand(min_weight, max_weight));
  }


}

/* deep copy of the fann structure */
  struct fann *FANN_API fann_copy(struct fann *orig) {
  struct fann *copy;
  unsigned int num_layers = (unsigned int)(orig->last_layer - orig->first_layer);
  struct fann_layer *orig_layer_it, *copy_layer_it;
  unsigned int layer_size;
  struct fann_neuron *last_neuron, *orig_neuron_it, *copy_neuron_it;
  unsigned int i;
  struct fann_neuron *orig_first_neuron, *copy_first_neuron;
  unsigned int input_neuron;

  copy = fann_allocate_structure(num_layers);
  if (copy == NULL) {
    fann_error((struct fann_error *)orig, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }
  copy->errno_f = orig->errno_f;
  if (orig->errstr) {
    copy->errstr = (char *)malloc(FANN_ERRSTR_MAX);
    if (copy->errstr == NULL) {
      fann_destroy(copy);
      return NULL;
    }
    strcpy(copy->errstr, orig->errstr);
  }
  copy->error_log = orig->error_log;

  copy->learning_rate = orig->learning_rate;
  copy->learning_momentum = orig->learning_momentum;
  copy->connection_rate = orig->connection_rate;
  copy->network_type = orig->network_type;
  copy->num_MSE = orig->num_MSE;
  copy->MSE_value = orig->MSE_value;
  copy->num_bit_fail = orig->num_bit_fail;
  copy->bit_fail_limit = orig->bit_fail_limit;
  copy->train_error_function = orig->train_error_function;
  copy->train_stop_function = orig->train_stop_function;
  copy->training_algorithm = orig->training_algorithm;
  copy->callback = orig->callback;
  copy->user_data = orig->user_data;

  copy->quickprop_decay = orig->quickprop_decay;
  copy->quickprop_mu = orig->quickprop_mu;
  copy->rprop_increase_factor = orig->rprop_increase_factor;
  copy->rprop_decrease_factor = orig->rprop_decrease_factor;
  copy->rprop_delta_min = orig->rprop_delta_min;
  copy->rprop_delta_max = orig->rprop_delta_max;
  copy->rprop_delta_zero = orig->rprop_delta_zero;

  /* user_data is not deep copied.  user should use fann_copy_with_user_data() for that */
  copy->user_data = orig->user_data;



  /* copy layer sizes, prepare for fann_allocate_neurons */
  for (orig_layer_it = orig->first_layer, copy_layer_it = copy->first_layer;
       orig_layer_it != orig->last_layer; orig_layer_it++, copy_layer_it++) {
    layer_size = (unsigned int)(orig_layer_it->last_neuron - orig_layer_it->first_neuron);
    copy_layer_it->first_neuron = NULL;
    copy_layer_it->last_neuron = copy_layer_it->first_neuron + layer_size;
    copy->total_neurons += layer_size;
  }
  copy->num_input = orig->num_input;
  copy->num_output = orig->num_output;

  /* copy scale parameters, when used */


  /* copy the neurons */
  fann_allocate_neurons(copy);
  if (copy->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
    fann_destroy(copy);
    return NULL;
  }
  layer_size =
      (unsigned int)((orig->last_layer - 1)->last_neuron - (orig->last_layer - 1)->first_neuron);
  memcpy(copy->output, orig->output, layer_size * sizeof(fann_type));

  last_neuron = (orig->last_layer - 1)->last_neuron;
  for (orig_neuron_it = orig->first_layer->first_neuron,
      copy_neuron_it = copy->first_layer->first_neuron;
       orig_neuron_it != last_neuron; orig_neuron_it++, copy_neuron_it++) {
    memcpy(copy_neuron_it, orig_neuron_it, sizeof(struct fann_neuron));
  }
  /* copy the connections */
  copy->total_connections = orig->total_connections;
  fann_allocate_connections(copy);
  if (copy->errno_f == FANN_E_CANT_ALLOCATE_MEM) {
    fann_destroy(copy);
    return NULL;
  }

  orig_first_neuron = orig->first_layer->first_neuron;
  copy_first_neuron = copy->first_layer->first_neuron;
  for (i = 0; i < orig->total_connections; i++) {
    copy->weights[i] = orig->weights[i];
    input_neuron = (unsigned int)(orig->connections[i] - orig_first_neuron);
    copy->connections[i] = copy_first_neuron + input_neuron;
  }

  if (orig->train_slopes) {
    copy->train_slopes = (fann_type *)malloc(copy->total_connections_allocated * sizeof(fann_type));
    if (copy->train_slopes == NULL) {
      fann_error((struct fann_error *)orig, FANN_E_CANT_ALLOCATE_MEM);
      fann_destroy(copy);
      return NULL;
    }
    memcpy(copy->train_slopes, orig->train_slopes,
           copy->total_connections_allocated * sizeof(fann_type));
  }

  if (orig->prev_steps) {
    copy->prev_steps = (fann_type *)malloc(copy->total_connections_allocated * sizeof(fann_type));
    if (copy->prev_steps == NULL) {
      fann_error((struct fann_error *)orig, FANN_E_CANT_ALLOCATE_MEM);
      fann_destroy(copy);
      return NULL;
    }
    memcpy(copy->prev_steps, orig->prev_steps,
           copy->total_connections_allocated * sizeof(fann_type));
  }

  if (orig->prev_train_slopes) {
    copy->prev_train_slopes =
        (fann_type *)malloc(copy->total_connections_allocated * sizeof(fann_type));
    if (copy->prev_train_slopes == NULL) {
      fann_error((struct fann_error *)orig, FANN_E_CANT_ALLOCATE_MEM);
      fann_destroy(copy);
      return NULL;
    }
    memcpy(copy->prev_train_slopes, orig->prev_train_slopes,
           copy->total_connections_allocated * sizeof(fann_type));
  }

  if (orig->prev_weights_deltas) {
    copy->prev_weights_deltas =
        (fann_type *)malloc(copy->total_connections_allocated * sizeof(fann_type));
    if (copy->prev_weights_deltas == NULL) {
      fann_error((struct fann_error *)orig, FANN_E_CANT_ALLOCATE_MEM);
      fann_destroy(copy);
      return NULL;
    }
    memcpy(copy->prev_weights_deltas, orig->prev_weights_deltas,
           copy->total_connections_allocated * sizeof(fann_type));
  }

  return copy;
}

  void FANN_API fann_print_connections(struct fann *ann) {
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it;
  unsigned int i;
  int value;
  char *neurons;
  unsigned int num_neurons = fann_get_total_neurons(ann) - fann_get_num_output(ann);

  neurons = (char *)malloc(num_neurons + 1);
  if (neurons == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }
  neurons[num_neurons] = 0;

  printf("Layer / Neuron ");
  for (i = 0; i < num_neurons; i++) {
    printf("%d", i % 10);
  }
  printf("\n");

  for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
    for (neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++) {
      memset(neurons, (int)'.', num_neurons);
      for (i = neuron_it->first_con; i < neuron_it->last_con; i++) {
        if (ann->weights[i] < 0) {

          value = (int)((ann->weights[i]) - 0.5);

          if (value < -25) value = -25;
          neurons[ann->connections[i] - ann->first_layer->first_neuron] = (char)('a' - value);
        } else {

          value = (int)((ann->weights[i]) + 0.5);

          if (value > 25) value = 25;
          neurons[ann->connections[i] - ann->first_layer->first_neuron] = (char)('A' + value);
        }
      }
      printf("L %3d / N %4d %s\n", (int)(layer_it - ann->first_layer),
             (int)(neuron_it - ann->first_layer->first_neuron), neurons);
    }
  }

  free(neurons);
}

/* Initialize the weights using Widrow + Nguyen's algorithm.
 */
  void FANN_API fann_init_weights(struct fann *ann,
                                              struct fann_train_data *train_data) {
  fann_type smallest_inp, largest_inp;
  unsigned int dat = 0, elem, num_connect, num_hidden_neurons;
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it, *last_neuron, *bias_neuron;


  float scale_factor;

  for (smallest_inp = largest_inp = train_data->input[0][0]; dat < train_data->num_data; dat++) {
    for (elem = 0; elem < train_data->num_input; elem++) {
      if (train_data->input[dat][elem] < smallest_inp) smallest_inp = train_data->input[dat][elem];
      if (train_data->input[dat][elem] > largest_inp) largest_inp = train_data->input[dat][elem];
    }
  }

  num_hidden_neurons = (unsigned int)(ann->total_neurons - (ann->num_input + ann->num_output +
                                                            (ann->last_layer - ann->first_layer)));
  scale_factor = (float)(pow((double)(0.7f * (double)num_hidden_neurons),
                             (double)(1.0f / (double)ann->num_input)) /
                         (double)(largest_inp - smallest_inp));


  bias_neuron = ann->first_layer->last_neuron - 1;
  for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
    last_neuron = layer_it->last_neuron;

    if (ann->network_type == FANN_NETTYPE_LAYER) {
      bias_neuron = (layer_it - 1)->last_neuron - 1;
    }

    for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
      for (num_connect = neuron_it->first_con; num_connect < neuron_it->last_con; num_connect++) {
        if (bias_neuron == ann->connections[num_connect]) {

          ann->weights[num_connect] = (fann_type)fann_rand(-scale_factor, scale_factor);

        } else {

          ann->weights[num_connect] = (fann_type)fann_rand(0, scale_factor);

        }
      }
    }
  }


}

  void FANN_API fann_print_parameters(struct fann *ann) {
  struct fann_layer *layer_it;


  printf("Input layer                          :%4d neurons, 1 bias\n", ann->num_input);
  for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer - 1; layer_it++) {
    if (ann->network_type == FANN_NETTYPE_SHORTCUT) {
      printf("  Hidden layer                       :%4d neurons, 0 bias\n",
             (int)(layer_it->last_neuron - layer_it->first_neuron));
    } else {
      printf("  Hidden layer                       :%4d neurons, 1 bias\n",
             (int)(layer_it->last_neuron - layer_it->first_neuron - 1));
    }
  }
  printf("Output layer                         :%4d neurons\n", ann->num_output);
  printf("Total neurons and biases             :%4d\n", fann_get_total_neurons(ann));
  printf("Total connections                    :%4d\n", ann->total_connections);
  printf("Connection rate                      :%8.3f\n", ann->connection_rate);
  printf("Network type                         :   %s\n", FANN_NETTYPE_NAMES[ann->network_type]);

  printf("Training algorithm                   :   %s\n",
         FANN_TRAIN_NAMES[ann->training_algorithm]);
  printf("Training error function              :   %s\n",
         FANN_ERRORFUNC_NAMES[ann->train_error_function]);
  printf("Training stop function               :   %s\n",
         FANN_STOPFUNC_NAMES[ann->train_stop_function]);


  printf("Bit fail limit                       :%8.3f\n", ann->bit_fail_limit);
  printf("Learning rate                        :%8.3f\n", ann->learning_rate);
  printf("Learning momentum                    :%8.3f\n", ann->learning_momentum);
  printf("Quickprop decay                      :%11.6f\n", ann->quickprop_decay);
  printf("Quickprop mu                         :%8.3f\n", ann->quickprop_mu);
  printf("RPROP increase factor                :%8.3f\n", ann->rprop_increase_factor);
  printf("RPROP decrease factor                :%8.3f\n", ann->rprop_decrease_factor);
  printf("RPROP delta min                      :%8.3f\n", ann->rprop_delta_min);
  printf("RPROP delta max                      :%8.3f\n", ann->rprop_delta_max);
  printf("Cascade output change fraction       :%11.6f\n", ann->cascade_output_change_fraction);
  printf("Cascade candidate change fraction    :%11.6f\n", ann->cascade_candidate_change_fraction);
  printf("Cascade output stagnation epochs     :%4d\n", ann->cascade_output_stagnation_epochs);
  printf("Cascade candidate stagnation epochs  :%4d\n", ann->cascade_candidate_stagnation_epochs);
  printf("Cascade max output epochs            :%4d\n", ann->cascade_max_out_epochs);
  printf("Cascade min output epochs            :%4d\n", ann->cascade_min_out_epochs);
  printf("Cascade max candidate epochs         :%4d\n", ann->cascade_max_cand_epochs);
  printf("Cascade min candidate epochs         :%4d\n", ann->cascade_min_cand_epochs);
  printf("Cascade weight multiplier            :%8.3f\n", ann->cascade_weight_multiplier);
  printf("Cascade candidate limit              :%8.3f\n", ann->cascade_candidate_limit);
  
  /* TODO: dump scale parameters */

}


  unsigned int FANN_API fann_get_total_neurons(struct fann *ann) {
  if (ann->network_type) {
    return ann->total_neurons;
  } else {
    /* -1, because there is always an unused bias neuron in the last layer */
    return ann->total_neurons - 1;
  }
}



  enum fann_nettype_enum FANN_API fann_get_network_type(struct fann *ann) {
  /* Currently two types: LAYER = 0, SHORTCUT = 1 */
  /* Enum network_types must be set to match the return values  */
  return ann->network_type;
}

  float FANN_API fann_get_connection_rate(struct fann *ann) {
  return ann->connection_rate;
}

  unsigned int FANN_API fann_get_num_layers(struct fann *ann) {
  return (unsigned int)(ann->last_layer - ann->first_layer);
}

  void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers) {
  struct fann_layer *layer_it;

  for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
    unsigned int count = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
    /* Remove the bias from the count of neurons. */
    switch (fann_get_network_type(ann)) {
      case FANN_NETTYPE_LAYER: {
        --count;
        break;
      }
      case FANN_NETTYPE_SHORTCUT: {
        /* The bias in the first layer is reused for all layers */
        if (layer_it == ann->first_layer) --count;
        break;
      }
      default: {
        /* Unknown network type, assume no bias present  */
        break;
      }
    }
    *layers++ = count;
  }
}

  void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias) {
  struct fann_layer *layer_it;

  for (layer_it = ann->first_layer; layer_it != ann->last_layer; ++layer_it, ++bias) {
    switch (fann_get_network_type(ann)) {
      case FANN_NETTYPE_LAYER: {
        /* Report one bias in each layer except the last */
        if (layer_it != ann->last_layer - 1)
          *bias = 1;
        else
          *bias = 0;
        break;
      }
      case FANN_NETTYPE_SHORTCUT: {
        /* The bias in the first layer is reused for all layers */
        if (layer_it == ann->first_layer)
          *bias = 1;
        else
          *bias = 0;
        break;
      }
      default: {
        /* Unknown network type, assume no bias present  */
        *bias = 0;
        break;
      }
    }
  }
}

  void FANN_API fann_get_connection_array(struct fann *ann,
                                                      struct fann_connection *connections) {
  struct fann_neuron *first_neuron;
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it;
  unsigned int idx;
  unsigned int source_index;
  unsigned int destination_index;

  first_neuron = ann->first_layer->first_neuron;

  source_index = 0;
  destination_index = 0;

  /* The following assumes that the last unused bias has no connections */

  /* for each layer */
  for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
    /* for each neuron */
    for (neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++) {
      /* for each connection */
      for (idx = neuron_it->first_con; idx < neuron_it->last_con; idx++) {
        /* Assign the source, destination and weight */
        connections->from_neuron = (unsigned int)(ann->connections[source_index] - first_neuron);
        connections->to_neuron = destination_index;
        connections->weight = ann->weights[source_index];

        connections++;
        source_index++;
      }
      destination_index++;
    }
  }
}

  void FANN_API fann_set_weight_array(struct fann *ann,
                                                  struct fann_connection *connections,
                                                  unsigned int num_connections) {
  unsigned int idx;

  for (idx = 0; idx < num_connections; idx++) {
    fann_set_weight(ann, connections[idx].from_neuron, connections[idx].to_neuron,
                    connections[idx].weight);
  }
}

  void FANN_API fann_set_weight(struct fann *ann, unsigned int from_neuron,
                                            unsigned int to_neuron, fann_type weight) {
  struct fann_neuron *first_neuron;
  struct fann_layer *layer_it;
  struct fann_neuron *neuron_it;
  unsigned int idx;
  unsigned int source_index;
  unsigned int destination_index;

  first_neuron = ann->first_layer->first_neuron;

  source_index = 0;
  destination_index = 0;

  /* Find the connection, simple brute force search through the network
     for one or more connections that match to minimize datastructure dependencies.
     Nothing is done if the connection does not already exist in the network. */

  /* for each layer */
  for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
    /* for each neuron */
    for (neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++) {
      /* for each connection */
      for (idx = neuron_it->first_con; idx < neuron_it->last_con; idx++) {
        /* If the source and destination neurons match, assign the weight */
        if (((int)from_neuron == ann->connections[source_index] - first_neuron) &&
            (to_neuron == destination_index)) {
          ann->weights[source_index] = weight;
        }
        source_index++;
      }
      destination_index++;
    }
  }
}

  void FANN_API fann_get_weights(struct fann *ann, fann_type *weights) {
  memcpy(weights, ann->weights, sizeof(fann_type) * ann->total_connections);
}

  void FANN_API fann_set_weights(struct fann *ann, fann_type *weights) {
  memcpy(ann->weights, weights, sizeof(fann_type) * ann->total_connections);
}


/* INTERNAL FUNCTION
   Allocates the main structure and sets some default values.
 */
struct fann *fann_allocate_structure(unsigned int num_layers) {
  struct fann *ann;

  if (num_layers < 2) {

    return NULL;
  }

  /* allocate and initialize the main network structure */
  ann = (struct fann *)malloc(sizeof(struct fann));
  if (ann == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  ann->errno_f = FANN_E_NO_ERROR;
  ann->error_log = fann_default_error_log;
  ann->errstr = NULL;
  ann->learning_rate = 0.7f;
  ann->learning_momentum = 0.0;
  ann->total_neurons = 0;
  ann->total_connections = 0;
  ann->num_input = 0;
  ann->num_output = 0;
  ann->train_errors = NULL;
  ann->train_slopes = NULL;
  ann->prev_steps = NULL;
  ann->prev_train_slopes = NULL;
  ann->prev_weights_deltas = NULL;
  ann->training_algorithm = FANN_TRAIN_RPROP;
  ann->num_MSE = 0;
  ann->MSE_value = 0;
  ann->num_bit_fail = 0;
  ann->bit_fail_limit = (fann_type)0.35;
  ann->network_type = FANN_NETTYPE_LAYER;
  ann->train_error_function = FANN_ERRORFUNC_TANH;
  ann->train_stop_function = FANN_STOPFUNC_MSE;
  ann->callback = NULL;
  ann->user_data = NULL; /* User is responsible for deallocation */
  ann->weights = NULL;
  ann->connections = NULL;
  ann->output = NULL;


  /* variables used for cascade correlation (reasonable defaults) */
  ann->cascade_output_change_fraction = 0.01f;
  ann->cascade_candidate_change_fraction = 0.01f;
  ann->cascade_output_stagnation_epochs = 12;
  ann->cascade_candidate_stagnation_epochs = 12;
  ann->cascade_num_candidate_groups = 2;
  ann->cascade_weight_multiplier = (fann_type)0.4;
  ann->cascade_candidate_limit = (fann_type)1000.0;
  ann->cascade_max_out_epochs = 150;
  ann->cascade_max_cand_epochs = 150;
  ann->cascade_min_out_epochs = 50;
  ann->cascade_min_cand_epochs = 50;
  ann->cascade_candidate_scores = NULL;
  ann->cascade_activation_functions_count = 10;
  ann->cascade_activation_functions = (enum fann_activationfunc_enum *)calloc(
      ann->cascade_activation_functions_count, sizeof(enum fann_activationfunc_enum));
  if (ann->cascade_activation_functions == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    free(ann);
    return NULL;
  }

  ann->cascade_activation_functions[0] = FANN_SIGMOID;
  ann->cascade_activation_functions[1] = FANN_SIGMOID_SYMMETRIC;
  ann->cascade_activation_functions[2] = FANN_GAUSSIAN;
  ann->cascade_activation_functions[3] = FANN_GAUSSIAN_SYMMETRIC;
  ann->cascade_activation_functions[4] = FANN_ELLIOT;
  ann->cascade_activation_functions[5] = FANN_ELLIOT_SYMMETRIC;
  ann->cascade_activation_functions[6] = FANN_SIN_SYMMETRIC;
  ann->cascade_activation_functions[7] = FANN_COS_SYMMETRIC;
  ann->cascade_activation_functions[8] = FANN_SIN;
  ann->cascade_activation_functions[9] = FANN_COS;

  ann->cascade_activation_steepnesses_count = 4;
  ann->cascade_activation_steepnesses =
      (fann_type *)calloc(ann->cascade_activation_steepnesses_count, sizeof(fann_type));
  if (ann->cascade_activation_steepnesses == NULL) {
    fann_safe_free(ann->cascade_activation_functions);
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    free(ann);
    return NULL;
  }

  ann->cascade_activation_steepnesses[0] = (fann_type)0.25;
  ann->cascade_activation_steepnesses[1] = (fann_type)0.5;
  ann->cascade_activation_steepnesses[2] = (fann_type)0.75;
  ann->cascade_activation_steepnesses[3] = (fann_type)1.0;

  /* Variables for use with with Quickprop training (reasonable defaults) */
  ann->quickprop_decay = -0.0001f;
  ann->quickprop_mu = 1.75;

  /* Variables for use with with RPROP training (reasonable defaults) */
  ann->rprop_increase_factor = 1.2f;
  ann->rprop_decrease_factor = 0.5;
  ann->rprop_delta_min = 0.0;
  ann->rprop_delta_max = 50.0;
  ann->rprop_delta_zero = 0.1f;

  /* Variables for use with SARPROP training (reasonable defaults) */
  ann->sarprop_weight_decay_shift = -6.644f;
  ann->sarprop_step_error_threshold_factor = 0.1f;
  ann->sarprop_step_error_shift = 1.385f;
  ann->sarprop_temperature = 0.015f;
  ann->sarprop_epoch = 0;

  fann_init_error_data((struct fann_error *)ann);



  /* allocate room for the layers */
  ann->first_layer = (struct fann_layer *)calloc(num_layers, sizeof(struct fann_layer));
  if (ann->first_layer == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    free(ann);
    return NULL;
  }

  ann->last_layer = ann->first_layer + num_layers;

  return ann;
}

/* INTERNAL FUNCTION
   Allocates room for the scaling parameters.
 */
int fann_allocate_scale(struct fann *ann) {
  /* todo this should only be allocated when needed */

  return 0;
}

/* INTERNAL FUNCTION
   Allocates room for the neurons.
 */
void fann_allocate_neurons(struct fann *ann) {
  struct fann_layer *layer_it;
  struct fann_neuron *neurons;
  unsigned int num_neurons_so_far = 0;
  unsigned int num_neurons = 0;

  /* all the neurons is allocated in one long array (calloc clears mem) */
  neurons = (struct fann_neuron *)calloc(ann->total_neurons, sizeof(struct fann_neuron));
  ann->total_neurons_allocated = ann->total_neurons;

  if (neurons == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }

  for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
    num_neurons = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
    layer_it->first_neuron = neurons + num_neurons_so_far;
    layer_it->last_neuron = layer_it->first_neuron + num_neurons;
    num_neurons_so_far += num_neurons;
  }

  ann->output = (fann_type *)calloc(num_neurons, sizeof(fann_type));
  if (ann->output == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }
}

/* INTERNAL FUNCTION
   Allocate room for the connections.
 */
void fann_allocate_connections(struct fann *ann) {
  ann->weights = (fann_type *)calloc(ann->total_connections, sizeof(fann_type));
  if (ann->weights == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }
  ann->total_connections_allocated = ann->total_connections;

  /* TODO make special cases for all places where the connections
   * is used, so that it is not needed for fully connected networks.
   */
  ann->connections =
      (struct fann_neuron **)calloc(ann->total_connections_allocated, sizeof(struct fann_neuron *));
  if (ann->connections == NULL) {
    fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
    return;
  }
}


int FANN_SEED_RAND = 1;


  void FANN_API fann_disable_seed_rand() { FANN_SEED_RAND = 0; }

  void FANN_API fann_enable_seed_rand() { FANN_SEED_RAND = 1; }

/* INTERNAL FUNCTION
   Seed the random function.
 */
void fann_seed_rand() {
#ifndef _WIN32
  FILE *fp = fopen("/dev/urandom", "r");
  unsigned int foo;
  struct timeval t;

  if (!fp) {
    gettimeofday(&t, NULL);
    foo = t.tv_usec;
#ifdef DEBUG
    printf("unable to open /dev/urandom\n");
#endif
  } else {
    if (fread(&foo, sizeof(foo), 1, fp) != 1) {
      gettimeofday(&t, NULL);
      foo = t.tv_usec;
#ifdef DEBUG
      printf("unable to read from /dev/urandom\n");
#endif
    }
    fclose(fp);
  }
  if (FANN_SEED_RAND) {
    srand(foo);
  }
#else
  /* COMPAT_TIME REPLACEMENT */
  if (FANN_SEED_RAND) {
    srand(GetTickCount());
  }
#endif
}

/* INTERNAL FUNCTION
   Helper function to update the MSE value and return a diff which takes symmetric functions into
   account
*/
fann_type fann_update_MSE(struct fann *ann, struct fann_neuron *neuron, fann_type neuron_diff) {
  float neuron_diff2;

  switch (neuron->activation_function) {
    case FANN_LINEAR_PIECE_SYMMETRIC:
    case FANN_THRESHOLD_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
    case FANN_ELLIOT_SYMMETRIC:
    case FANN_GAUSSIAN_SYMMETRIC:
    case FANN_SIN_SYMMETRIC:
    case FANN_COS_SYMMETRIC:
      neuron_diff /= (fann_type)2.0;
      break;
    case FANN_THRESHOLD:
    case FANN_LINEAR:
    case FANN_SIGMOID:
    case FANN_SIGMOID_STEPWISE:
    case FANN_GAUSSIAN:
    case FANN_GAUSSIAN_STEPWISE:
    case FANN_ELLIOT:
    case FANN_LINEAR_PIECE:
    case FANN_SIN:
    case FANN_COS:
      break;
  }


  neuron_diff2 = (float)(neuron_diff * neuron_diff);


  ann->MSE_value += neuron_diff2;

  /*printf("neuron_diff %f = (%f - %f)[/2], neuron_diff2=%f, sum=%f, MSE_value=%f, num_MSE=%d\n",
   * neuron_diff, *desired_output, neuron_value, neuron_diff2, last_layer_begin->sum,
   * ann->MSE_value, ann->num_MSE); */
  if (fann_abs(neuron_diff) >= ann->bit_fail_limit) {
    ann->num_bit_fail++;
  }

  return neuron_diff;
}

/* Tests the network.
 */
  fann_type *FANN_API fann_test(struct fann *ann, fann_type *input,
                                            fann_type *desired_output) {
  fann_type neuron_value;
  fann_type *output_begin = fann_run(ann, input);
  fann_type *output_it;
  const fann_type *output_end = output_begin + ann->num_output;
  fann_type neuron_diff;
  struct fann_neuron *output_neuron = (ann->last_layer - 1)->first_neuron;

  /* calculate the error */
  for (output_it = output_begin; output_it != output_end; output_it++) {
    neuron_value = *output_it;

    neuron_diff = (*desired_output - neuron_value);

    neuron_diff = fann_update_MSE(ann, output_neuron, neuron_diff);

    desired_output++;
    output_neuron++;

    ann->num_MSE++;
  }

  return output_begin;
}

/* get the mean square error.
 */
  float FANN_API fann_get_MSE(struct fann *ann) {
  if (ann->num_MSE) {
    return ann->MSE_value / (float)ann->num_MSE;
  } else {
    return 0;
  }
}

  unsigned int FANN_API fann_get_bit_fail(struct fann *ann) {
  return ann->num_bit_fail;
}

/* reset the mean square error.
 */
  void FANN_API fann_reset_MSE(struct fann *ann) {
  /*printf("resetMSE %d %f\n", ann->num_MSE, ann->MSE_value);*/
  ann->num_MSE = 0;
  ann->MSE_value = 0;
  ann->num_bit_fail = 0;
}


  void FANN_API fann_set_activation_function_hidden(
    struct fann *ann, enum fann_activationfunc_enum activation_function) {
  struct fann_neuron *last_neuron, *neuron_it;
  struct fann_layer *layer_it;
  struct fann_layer *last_layer = ann->last_layer - 1; /* -1 to not update the output layer */

  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
    last_neuron = layer_it->last_neuron;
    for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
      neuron_it->activation_function = activation_function;
    }
  }
}

  struct fann_layer *FANN_API fann_get_layer(struct fann *ann, int layer) {
  if (layer <= 0 || layer >= (ann->last_layer - ann->first_layer)) {
    fann_error((struct fann_error *)ann, FANN_E_INDEX_OUT_OF_BOUND, layer);
    return NULL;
  }

  return ann->first_layer + layer;
}

  struct fann_neuron *FANN_API fann_get_neuron_layer(struct fann *ann,
                                                                 struct fann_layer *layer,
                                                                 int neuron) {
  if (neuron >= (layer->last_neuron - layer->first_neuron)) {
    fann_error((struct fann_error *)ann, FANN_E_INDEX_OUT_OF_BOUND, neuron);
    return NULL;
  }

  return layer->first_neuron + neuron;
}

  struct fann_neuron *FANN_API fann_get_neuron(struct fann *ann, unsigned int layer,
                                                           int neuron) {
  struct fann_layer *layer_it = fann_get_layer(ann, layer);
  if (layer_it == NULL) return NULL;
  return fann_get_neuron_layer(ann, layer_it, neuron);
}

  enum fann_activationfunc_enum FANN_API fann_get_activation_function(struct fann *ann,
                                                                                  int layer,
                                                                                  int neuron) {
  struct fann_neuron *neuron_it = fann_get_neuron(ann, layer, neuron);
  if (neuron_it == NULL) {
    return (enum fann_activationfunc_enum) - 1; /* layer or neuron out of bounds */
  } else {
    return neuron_it->activation_function;
  }
}

  void FANN_API fann_set_activation_function(
    struct fann *ann, enum fann_activationfunc_enum activation_function, int layer, int neuron) {
  struct fann_neuron *neuron_it = fann_get_neuron(ann, layer, neuron);
  if (neuron_it == NULL) return;

  neuron_it->activation_function = activation_function;
}

  void FANN_API fann_set_activation_function_layer(
    struct fann *ann, enum fann_activationfunc_enum activation_function, int layer) {
  struct fann_neuron *last_neuron, *neuron_it;
  struct fann_layer *layer_it = fann_get_layer(ann, layer);

  if (layer_it == NULL) return;

  last_neuron = layer_it->last_neuron;
  for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
    neuron_it->activation_function = activation_function;
  }
}

  void FANN_API fann_set_activation_function_output(
    struct fann *ann, enum fann_activationfunc_enum activation_function) {
  struct fann_neuron *last_neuron, *neuron_it;
  struct fann_layer *last_layer = ann->last_layer - 1;

  last_neuron = last_layer->last_neuron;
  for (neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++) {
    neuron_it->activation_function = activation_function;
  }
}

  void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
                                                                 fann_type steepness) {
  struct fann_neuron *last_neuron, *neuron_it;
  struct fann_layer *layer_it;
  struct fann_layer *last_layer = ann->last_layer - 1; /* -1 to not update the output layer */

  for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
    last_neuron = layer_it->last_neuron;
    for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
      neuron_it->activation_steepness = steepness;
    }
  }
}

  fann_type FANN_API fann_get_activation_steepness(struct fann *ann, int layer,
                                                               int neuron) {
  struct fann_neuron *neuron_it = fann_get_neuron(ann, layer, neuron);
  if (neuron_it == NULL) {
    return -1; /* layer or neuron out of bounds */
  } else {
    return neuron_it->activation_steepness;
  }
}

  void FANN_API fann_set_activation_steepness(struct fann *ann, fann_type steepness,
                                                          int layer, int neuron) {
  struct fann_neuron *neuron_it = fann_get_neuron(ann, layer, neuron);
  if (neuron_it == NULL) return;

  neuron_it->activation_steepness = steepness;
}

  void FANN_API fann_set_activation_steepness_layer(struct fann *ann,
                                                                fann_type steepness, int layer) {
  struct fann_neuron *last_neuron, *neuron_it;
  struct fann_layer *layer_it = fann_get_layer(ann, layer);

  if (layer_it == NULL) return;

  last_neuron = layer_it->last_neuron;
  for (neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
    neuron_it->activation_steepness = steepness;
  }
}

  void FANN_API fann_set_activation_steepness_output(struct fann *ann,
                                                                 fann_type steepness) {
  struct fann_neuron *last_neuron, *neuron_it;
  struct fann_layer *last_layer = ann->last_layer - 1;

  last_neuron = last_layer->last_neuron;
  for (neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++) {
    neuron_it->activation_steepness = steepness;
  }
}


/*
 * Reads training data from a file.
 */
  struct fann_train_data *FANN_API
fann_read_train_from_file(const char *configuration_file) {
  struct fann_train_data *data;
  FILE *file = fopen(configuration_file, "r");

  if (!file) {
    fann_error(NULL, FANN_E_CANT_OPEN_CONFIG_R, configuration_file);
    return NULL;
  }

  data = fann_read_train_from_fd(file, configuration_file);
  fclose(file);
  return data;
}

/*
 * Save training data to a file
 */
  int FANN_API fann_save_train(struct fann_train_data *data, const char *filename) {
  return fann_save_train_internal(data, filename, 0, 0);
}

/*
 * Save training data to a file in fixed point algebra. (Good for testing
 * a network in fixed point)
 */
  int FANN_API fann_save_train_to_fixed(struct fann_train_data *data,
                                                    const char *filename,
                                                    unsigned int decimal_point) {
  return fann_save_train_internal(data, filename, 1, decimal_point);
}

/*
 * deallocate the train data structure.
 */
  void FANN_API fann_destroy_train(struct fann_train_data *data) {
  if (data == NULL) return;
  if (data->input != NULL) fann_safe_free(data->input[0]);
  if (data->output != NULL) fann_safe_free(data->output[0]);
  fann_safe_free(data->input);
  fann_safe_free(data->output);
  fann_safe_free(data);
}

/*
 * Test a set of training data and calculate the MSE
 */
  float FANN_API fann_test_data(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;
  if (fann_check_input_output_sizes(ann, data) == -1) return 0;

  fann_reset_MSE(ann);

  for (i = 0; i != data->num_data; i++) {
    fann_test(ann, data->input[i], data->output[i]);
  }

  return fann_get_MSE(ann);
}


/*
 * shuffles training data, randomizing the order
 */
  void FANN_API fann_shuffle_train_data(struct fann_train_data *train_data) {
  unsigned int dat = 0, elem, swap;
  fann_type temp;

  for (; dat < train_data->num_data; dat++) {
    swap = (unsigned int)(rand() % train_data->num_data);
    if (swap != dat) {
      for (elem = 0; elem < train_data->num_input; elem++) {
        temp = train_data->input[dat][elem];
        train_data->input[dat][elem] = train_data->input[swap][elem];
        train_data->input[swap][elem] = temp;
      }
      for (elem = 0; elem < train_data->num_output; elem++) {
        temp = train_data->output[dat][elem];
        train_data->output[dat][elem] = train_data->output[swap][elem];
        train_data->output[swap][elem] = temp;
      }
    }
  }
}

/*
 * INTERNAL FUNCTION calculates min and max of train data
 */
void fann_get_min_max_data(fann_type **data, unsigned int num_data, unsigned int num_elem,
                           fann_type *min, fann_type *max) {
  fann_type temp;
  unsigned int dat, elem;
  *min = *max = data[0][0];

  for (dat = 0; dat < num_data; dat++) {
    for (elem = 0; elem < num_elem; elem++) {
      temp = data[dat][elem];
      if (temp < *min)
        *min = temp;
      else if (temp > *max)
        *max = temp;
    }
  }
}

  fann_type FANN_API fann_get_min_train_input(struct fann_train_data *train_data) {
  fann_type min, max;
  fann_get_min_max_data(train_data->input, train_data->num_data, train_data->num_input, &min, &max);
  return min;
}

  fann_type FANN_API fann_get_max_train_input(struct fann_train_data *train_data) {
  fann_type min, max;
  fann_get_min_max_data(train_data->input, train_data->num_data, train_data->num_input, &min, &max);
  return max;
}

  fann_type FANN_API fann_get_min_train_output(struct fann_train_data *train_data) {
  fann_type min, max;
  fann_get_min_max_data(train_data->output, train_data->num_data, train_data->num_output, &min,
                        &max);
  return min;
}

  fann_type FANN_API fann_get_max_train_output(struct fann_train_data *train_data) {
  fann_type min, max;
  fann_get_min_max_data(train_data->output, train_data->num_data, train_data->num_output, &min,
                        &max);
  return max;
}

/*
 * INTERNAL FUNCTION Scales data to a specific range
 */
void fann_scale_data(fann_type **data, unsigned int num_data, unsigned int num_elem,
                     fann_type new_min, fann_type new_max) {
  fann_type old_min, old_max;
  fann_get_min_max_data(data, num_data, num_elem, &old_min, &old_max);
  fann_scale_data_to_range(data, num_data, num_elem, old_min, old_max, new_min, new_max);
}

/*
 * INTERNAL FUNCTION Scales data to a specific range
 */
  void FANN_API fann_scale_data_to_range(fann_type **data, unsigned int num_data,
                                                     unsigned int num_elem, fann_type old_min,
                                                     fann_type old_max, fann_type new_min,
                                                     fann_type new_max) {
  unsigned int dat, elem;
  fann_type temp, old_span, new_span, factor;

  old_span = old_max - old_min;
  new_span = new_max - new_min;
  factor = new_span / old_span;
  /*printf("max %f, min %f, factor %f\n", old_max, old_min, factor);*/

  for (dat = 0; dat < num_data; dat++) {
    for (elem = 0; elem < num_elem; elem++) {
      temp = (data[dat][elem] - old_min) * factor + new_min;
      if (temp < new_min) {
        data[dat][elem] = new_min;
        /*
         * printf("error %f < %f\n", temp, new_min);
         */
      } else if (temp > new_max) {
        data[dat][elem] = new_max;
        /*
         * printf("error %f > %f\n", temp, new_max);
         */
      } else {
        data[dat][elem] = temp;
      }
    }
  }
}

/*
 * Scales the inputs in the training data to the specified range
 */
  void FANN_API fann_scale_input_train_data(struct fann_train_data *train_data,
                                                        fann_type new_min, fann_type new_max) {
  fann_scale_data(train_data->input, train_data->num_data, train_data->num_input, new_min, new_max);
}

/*
 * Scales the inputs in the training data to the specified range
 */
  void FANN_API fann_scale_output_train_data(struct fann_train_data *train_data,
                                                         fann_type new_min, fann_type new_max) {
  fann_scale_data(train_data->output, train_data->num_data, train_data->num_output, new_min,
                  new_max);
}

/*
 * Scales the inputs in the training data to the specified range
 */
  void FANN_API fann_scale_train_data(struct fann_train_data *train_data,
                                                  fann_type new_min, fann_type new_max) {
  fann_scale_data(train_data->input, train_data->num_data, train_data->num_input, new_min, new_max);
  fann_scale_data(train_data->output, train_data->num_data, train_data->num_output, new_min,
                  new_max);
}

/*
 * merges training data into a single struct.
 */
  struct fann_train_data *FANN_API
fann_merge_train_data(struct fann_train_data *data1, struct fann_train_data *data2) {
  unsigned int i;
  fann_type *data_input, *data_output;
  struct fann_train_data *dest = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

  if (dest == NULL) {
    fann_error((struct fann_error *)data1, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  if ((data1->num_input != data2->num_input) || (data1->num_output != data2->num_output)) {
    fann_error((struct fann_error *)data1, FANN_E_TRAIN_DATA_MISMATCH);
    return NULL;
  }

  fann_init_error_data((struct fann_error *)dest);
  dest->error_log = data1->error_log;

  dest->num_data = data1->num_data + data2->num_data;
  dest->num_input = data1->num_input;
  dest->num_output = data1->num_output;
  dest->input = (fann_type **)calloc(dest->num_data, sizeof(fann_type *));
  if (dest->input == NULL) {
    fann_error((struct fann_error *)data1, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }

  dest->output = (fann_type **)calloc(dest->num_data, sizeof(fann_type *));
  if (dest->output == NULL) {
    fann_error((struct fann_error *)data1, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }

  data_input = (fann_type *)calloc(dest->num_input * dest->num_data, sizeof(fann_type));
  if (data_input == NULL) {
    fann_error((struct fann_error *)data1, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }
  memcpy(data_input, data1->input[0], dest->num_input * data1->num_data * sizeof(fann_type));
  memcpy(data_input + (dest->num_input * data1->num_data), data2->input[0],
         dest->num_input * data2->num_data * sizeof(fann_type));

  data_output = (fann_type *)calloc(dest->num_output * dest->num_data, sizeof(fann_type));
  if (data_output == NULL) {
    fann_error((struct fann_error *)data1, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }
  memcpy(data_output, data1->output[0], dest->num_output * data1->num_data * sizeof(fann_type));
  memcpy(data_output + (dest->num_output * data1->num_data), data2->output[0],
         dest->num_output * data2->num_data * sizeof(fann_type));

  for (i = 0; i != dest->num_data; i++) {
    dest->input[i] = data_input;
    data_input += dest->num_input;
    dest->output[i] = data_output;
    data_output += dest->num_output;
  }
  return dest;
}

/*
 * return a copy of a fann_train_data struct
 */
  struct fann_train_data *FANN_API
fann_duplicate_train_data(struct fann_train_data *data) {
  unsigned int i;
  fann_type *data_input, *data_output;
  struct fann_train_data *dest = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

  if (dest == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  fann_init_error_data((struct fann_error *)dest);
  dest->error_log = data->error_log;

  dest->num_data = data->num_data;
  dest->num_input = data->num_input;
  dest->num_output = data->num_output;
  dest->input = (fann_type **)calloc(dest->num_data, sizeof(fann_type *));
  if (dest->input == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }

  dest->output = (fann_type **)calloc(dest->num_data, sizeof(fann_type *));
  if (dest->output == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }

  data_input = (fann_type *)calloc(dest->num_input * dest->num_data, sizeof(fann_type));
  if (data_input == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }
  memcpy(data_input, data->input[0], dest->num_input * dest->num_data * sizeof(fann_type));

  data_output = (fann_type *)calloc(dest->num_output * dest->num_data, sizeof(fann_type));
  if (data_output == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }
  memcpy(data_output, data->output[0], dest->num_output * dest->num_data * sizeof(fann_type));

  for (i = 0; i != dest->num_data; i++) {
    dest->input[i] = data_input;
    data_input += dest->num_input;
    dest->output[i] = data_output;
    data_output += dest->num_output;
  }
  return dest;
}

  struct fann_train_data *FANN_API fann_subset_train_data(struct fann_train_data *data,
                                                                      unsigned int pos,
                                                                      unsigned int length) {
  unsigned int i;
  fann_type *data_input, *data_output;
  struct fann_train_data *dest = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

  if (dest == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  if (pos > data->num_data || pos + length > data->num_data) {
    fann_error((struct fann_error *)data, FANN_E_TRAIN_DATA_SUBSET, pos, length, data->num_data);
    return NULL;
  }

  fann_init_error_data((struct fann_error *)dest);
  dest->error_log = data->error_log;

  dest->num_data = length;
  dest->num_input = data->num_input;
  dest->num_output = data->num_output;
  dest->input = (fann_type **)calloc(dest->num_data, sizeof(fann_type *));
  if (dest->input == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }

  dest->output = (fann_type **)calloc(dest->num_data, sizeof(fann_type *));
  if (dest->output == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }

  data_input = (fann_type *)calloc(dest->num_input * dest->num_data, sizeof(fann_type));
  if (data_input == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }
  memcpy(data_input, data->input[pos], dest->num_input * dest->num_data * sizeof(fann_type));

  data_output = (fann_type *)calloc(dest->num_output * dest->num_data, sizeof(fann_type));
  if (data_output == NULL) {
    fann_error((struct fann_error *)data, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(dest);
    return NULL;
  }
  memcpy(data_output, data->output[pos], dest->num_output * dest->num_data * sizeof(fann_type));

  for (i = 0; i != dest->num_data; i++) {
    dest->input[i] = data_input;
    data_input += dest->num_input;
    dest->output[i] = data_output;
    data_output += dest->num_output;
  }
  return dest;
}

  unsigned int FANN_API fann_length_train_data(struct fann_train_data *data) {
  return data->num_data;
}

  unsigned int FANN_API fann_num_input_train_data(struct fann_train_data *data) {
  return data->num_input;
}

  unsigned int FANN_API fann_num_output_train_data(struct fann_train_data *data) {
  return data->num_output;
}

/* INTERNAL FUNCTION
   Save the train data structure.
 */
int fann_save_train_internal(struct fann_train_data *data, const char *filename,
                             unsigned int save_as_fixed, unsigned int decimal_point) {
  int retval = 0;
  FILE *file = fopen(filename, "w");

  if (!file) {
    fann_error((struct fann_error *)data, FANN_E_CANT_OPEN_TD_W, filename);
    return -1;
  }
  retval = fann_save_train_internal_fd(data, file, filename, save_as_fixed, decimal_point);
  fclose(file);

  return retval;
}

# define FANNPRINTF "%.20e"
/* INTERNAL FUNCTION
   Save the train data structure.
 */
int fann_save_train_internal_fd(struct fann_train_data *data, FILE *file, const char *filename,
                                unsigned int save_as_fixed, unsigned int decimal_point) {
  unsigned int num_data = data->num_data;
  unsigned int num_input = data->num_input;
  unsigned int num_output = data->num_output;
  unsigned int i, j;
  int retval = 0;


  fprintf(file, "%u %u %u\n", data->num_data, data->num_input, data->num_output);

  for (i = 0; i < num_data; i++) {
    for (j = 0; j < num_input; j++) {

      fprintf(file, FANNPRINTF " ", data->input[i][j]);

    }
    fprintf(file, "\n");

    for (j = 0; j < num_output; j++) {

      fprintf(file, FANNPRINTF " ", data->output[i][j]);

    }
    fprintf(file, "\n");
  }

  return retval;
}

/*
 * Creates an empty set of training data
 */
  struct fann_train_data *FANN_API fann_create_train(unsigned int num_data,
                                                                 unsigned int num_input,
                                                                 unsigned int num_output) {
  fann_type *data_input, *data_output;
  unsigned int i;
  struct fann_train_data *data = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

  if (data == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    return NULL;
  }

  fann_init_error_data((struct fann_error *)data);

  data->num_data = num_data;
  data->num_input = num_input;
  data->num_output = num_output;
  data->input = (fann_type **)calloc(num_data, sizeof(fann_type *));
  if (data->input == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }

  data->output = (fann_type **)calloc(num_data, sizeof(fann_type *));
  if (data->output == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }

  data_input = (fann_type *)calloc(num_input * num_data, sizeof(fann_type));
  if (data_input == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }

  data_output = (fann_type *)calloc(num_output * num_data, sizeof(fann_type));
  if (data_output == NULL) {
    fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
    fann_destroy_train(data);
    return NULL;
  }

  for (i = 0; i != num_data; i++) {
    data->input[i] = data_input;
    data_input += num_input;
    data->output[i] = data_output;
    data_output += num_output;
  }
  return data;
}

  struct fann_train_data *FANN_API
fann_create_train_pointer_array(unsigned int num_data, unsigned int num_input, fann_type **input,
                                unsigned int num_output, fann_type **output) {
  unsigned int i;
  struct fann_train_data *data;
  data = fann_create_train(num_data, num_input, num_output);

  if (data == NULL) return NULL;

  for (i = 0; i < num_data; ++i) {
    memcpy(data->input[i], input[i], num_input * sizeof(fann_type));
    memcpy(data->output[i], output[i], num_output * sizeof(fann_type));
  }

  return data;
}

  struct fann_train_data *FANN_API fann_create_train_array(unsigned int num_data,
                                                                       unsigned int num_input,
                                                                       fann_type *input,
                                                                       unsigned int num_output,
                                                                       fann_type *output) {
  unsigned int i;
  struct fann_train_data *data;
  data = fann_create_train(num_data, num_input, num_output);

  if (data == NULL) return NULL;

  for (i = 0; i < num_data; ++i) {
    memcpy(data->input[i], &input[i * num_input], num_input * sizeof(fann_type));
    memcpy(data->output[i], &output[i * num_output], num_output * sizeof(fann_type));
  }

  return data;
}

/*
 * Creates training data from a callback function.
 */
  struct fann_train_data *FANN_API fann_create_train_from_callback(
    unsigned int num_data, unsigned int num_input, unsigned int num_output,
    void(FANN_API *user_function)(unsigned int, unsigned int, unsigned int, fann_type *,
                                  fann_type *)) {
  unsigned int i;
  struct fann_train_data *data = fann_create_train(num_data, num_input, num_output);
  if (data == NULL) {
    return NULL;
  }

  for (i = 0; i != num_data; i++) {
    (*user_function)(i, num_input, num_output, data->input[i], data->output[i]);
  }

  return data;
}

  fann_type *FANN_API fann_get_train_input(struct fann_train_data *data,
                                                       unsigned int position) {
  if (position >= data->num_data) return NULL;
  return data->input[position];
}

  fann_type *FANN_API fann_get_train_output(struct fann_train_data *data,
                                                        unsigned int position) {
  if (position >= data->num_data) return NULL;
  return data->output[position];
}

#define FANNSCANF "%le"

/*
 * INTERNAL FUNCTION Reads training data from a file descriptor.
 */

struct fann_train_data *fann_read_train_from_fd(FILE *file, const char *filename) {
  unsigned int num_input, num_output, num_data, i, j;
  unsigned int line = 1;
  struct fann_train_data *data;

  if (fscanf(file, "%u %u %u\n", &num_data, &num_input, &num_output) != 3) {
    fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
    return NULL;
  }
  line++;

  data = fann_create_train(num_data, num_input, num_output);
  if (data == NULL) {
    return NULL;
  }

  for (i = 0; i != num_data; i++) {
    for (j = 0; j != num_input; j++) {
      if (fscanf(file, FANNSCANF " ", &data->input[i][j]) != 1) {
        fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
        fann_destroy_train(data);
        return NULL;
      }
    }
    line++;

    for (j = 0; j != num_output; j++) {
      if (fscanf(file, FANNSCANF " ", &data->output[i][j]) != 1) {
        fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
        fann_destroy_train(data);
        return NULL;
      }
    }
    line++;
  }
  return data;
}

/*
 * INTERNAL FUNCTION returns 0 if the desired error is reached and -1 if it is not reached
 */
int fann_desired_error_reached(struct fann *ann, float desired_error) {
  switch (ann->train_stop_function) {
    case FANN_STOPFUNC_MSE:
      if (fann_get_MSE(ann) <= desired_error) return 0;
      break;
    case FANN_STOPFUNC_BIT:
      if (ann->num_bit_fail <= (unsigned int)desired_error) return 0;
      break;
  }
  return -1;
}


int fann_check_input_output_sizes(struct fann *ann, struct fann_train_data *data) {
  if (ann->num_input != data->num_input) {
    fann_error((struct fann_error *)ann, FANN_E_INPUT_NO_MATCH, ann->num_input, data->num_input);
    return -1;
  }

  if (ann->num_output != data->num_output) {
    fann_error((struct fann_error *)ann, FANN_E_OUTPUT_NO_MATCH, ann->num_output, data->num_output);
    return -1;
  }

  return 0;
}

float fann_train_epoch_quickprop(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;

  if (ann->prev_train_slopes == NULL) {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  for (i = 0; i < data->num_data; i++) {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }
  fann_update_weights_quickprop(ann, data->num_data, 0, ann->total_connections);

  return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_train_epoch_irpropm(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;

  if (ann->prev_train_slopes == NULL) {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);
  
  for (i = 0; i < data->num_data; i++) {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }

  fann_update_weights_irpropm(ann, 0, ann->total_connections);

  return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_train_epoch_sarprop(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;

  if (ann->prev_train_slopes == NULL) {
    fann_clear_train_arrays(ann);
  }

  fann_reset_MSE(ann);

  for (i = 0; i < data->num_data; i++) {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }

  fann_update_weights_sarprop(ann, ann->sarprop_epoch, 0, ann->total_connections);

  ++(ann->sarprop_epoch);

  return fann_get_MSE(ann);
}

/*
 * Internal train function
 */
float fann_train_epoch_batch(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;

  fann_reset_MSE(ann);

  for (i = 0; i < data->num_data; i++) {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }

  fann_update_weights_batch(ann, data->num_data, 0, ann->total_connections);

  return fann_get_MSE(ann);
}

void FANN_API fann_train(struct fann *ann, fann_type *input,
                                       fann_type *desired_output) {
  fann_run(ann, input);

  fann_compute_MSE(ann, desired_output);

  fann_backpropagate_MSE(ann);
  
  fann_update_weights(ann);
}
/*
 * Internal train function
 */
float fann_train_epoch_incremental(struct fann *ann, struct fann_train_data *data) {
  unsigned int i;

  fann_reset_MSE(ann);

  for (i = 0; i != data->num_data; i++) {
    fann_train(ann, data->input[i], data->output[i]);
  }

  return fann_get_MSE(ann);
}

/*
 * Train for one epoch with the selected training algorithm
 */
float FANN_API fann_train_epoch(struct fann *ann, struct fann_train_data *data) {
  if (fann_check_input_output_sizes(ann, data) == -1) return 0;

  switch (ann->training_algorithm) {
    case FANN_TRAIN_QUICKPROP:
      return fann_train_epoch_quickprop(ann, data);
    case FANN_TRAIN_RPROP:
      return fann_train_epoch_irpropm(ann, data);
    case FANN_TRAIN_SARPROP:
      return fann_train_epoch_sarprop(ann, data);
    case FANN_TRAIN_BATCH:
      return fann_train_epoch_batch(ann, data);
    case FANN_TRAIN_INCREMENTAL:
      return fann_train_epoch_incremental(ann, data);
  }
  return 0;
}

void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
                                               unsigned int max_epochs,
                                               unsigned int epochs_between_reports,
                                               float desired_error) {
 
  float error;
  unsigned int i=0;
  int desired_error_reached;



  if (epochs_between_reports && ann->callback == NULL) {
    printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);
  }

  for (i = 1; i <= max_epochs; i++) {
    /*
     * train
     */
    error = fann_train_epoch(ann, data);
    desired_error_reached = fann_desired_error_reached(ann, desired_error);

    /*
     * print current output
     */
    if (epochs_between_reports && (i % epochs_between_reports == 0 || i == max_epochs || i == 1 ||
                                   desired_error_reached == 0)) {
      if (ann->callback == NULL) {
        printf("Epochs     %8d. Current error: %.10f. Bit fail %d.\n", i, error, ann->num_bit_fail);
      } else if (((*ann->callback)(ann, data, max_epochs, epochs_between_reports, desired_error,
                                   i)) == -1) {
        /*
         * you can break the training by returning -1
         */
        break;
      }
    }

    if (desired_error_reached == 0) break;
  }
}

void FANN_API fann_set_train_stop_function(struct fann *ann, enum fann_stopfunc_enum stopfunction)
{
  ann->train_stop_function=stopfunction;
}

void FANN_API fann_set_bit_fail_limit(struct fann *ann, fann_type bit_fail)
{
  ann->bit_fail_limit=bit_fail;
}

void FANN_API fann_set_training_algorithm(struct fann *ann, enum fann_train_enum training_algo)
{
  ann->training_algorithm=training_algo;
}

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports, 
	float desired_error, unsigned int epochs)
{
	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

int main( int argc, char* argv[])
{
fann_type *calc_out;
	unsigned int num_input;
	unsigned int num_output;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = 0.001;
	const unsigned int max_epochs = 30;
	const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *data;
  struct fann_train_data *test_data;

	unsigned int i = 0;
	
  data = fann_read_train_from_file(argv[1]);
  num_input=data->num_input;
  num_output=data->num_output;
	printf("Creating network.\n");
 
	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
 
	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	fann_init_weights(ann, data);
	
	printf("Training network.\n");
  fann_print_parameters(ann);
  fann_print_connections(ann);

	fann_train_on_data(ann,data,max_epochs,epochs_between_reports,desired_error);

	fann_destroy_train(data);

	int ret = 0;

	printf("Creating network.\n");

	if(!ann)
	{
		printf("Error creating ann --- ABORTING.\n");
		return -1;
	}

	printf("Testing network.\n");

	test_data = fann_read_train_from_file((argv[2]));

  for(i = 0; i < fann_length_train_data(test_data); i++)
	{
		fann_reset_MSE(ann);
		calc_out = fann_test(ann, test_data->input[i], test_data->output[i]);

		printf("Test (%f, %f) -> %f, should be %f, difference=%f\n",
			   test_data->input[i][0], test_data->input[i][1], calc_out[0], test_data->output[i][0],
			   (float) fann_abs(calc_out[0] - test_data->output[i][0]));
  }
  printf("MSE value for the test data (%f)\n", fann_get_MSE(ann));
	printf("Cleaning up.\n");
	fann_destroy_train(test_data);
	fann_destroy(ann);
  
	return ret;
}
