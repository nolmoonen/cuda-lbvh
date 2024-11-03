#pragma once

typedef unsigned int uint;
typedef unsigned char uchar;

/// Divide N by S, round up result.
#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)))
