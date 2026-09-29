#include "nn.h"

const char *nn_status_str(nn_status status)
{
    switch (status) {
    case NN_OK:                return "ok";
    case NN_ERR_NULL_ARG:      return "null argument";
    case NN_ERR_INVALID_ARG:   return "invalid argument";
    case NN_ERR_ALLOC:         return "memory allocation failed";
    case NN_ERR_SIZE_MISMATCH: return "size mismatch";
    case NN_ERR_IO:            return "I/O error";
    case NN_ERR_FORMAT:        return "malformed file";
    }
    return "unknown status";
}
