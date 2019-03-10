#include <stdint.h>

typedef uint8_t  BYTE;
typedef uint32_t DWORD;
typedef uint64_t HEADER;
typedef int32_t  LONG;
typedef uint16_t WORD;


typedef struct
{
    float value[8];
} __attribute__((__packed__))
DATASET;
