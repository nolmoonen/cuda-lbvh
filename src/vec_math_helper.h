#pragma once

#include "../deps/sutil/vec_math.h"

SUTIL_INLINE SUTIL_HOSTDEVICE float2 fabsf(const float2 &v)
{ return make_float2(fabsf(v.x), fabsf(v.y)); }

SUTIL_INLINE SUTIL_HOSTDEVICE float3 fabsf(const float3 &v)
{ return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }

/// GLSL mod (https://docs.gl/sl4/mod)
SUTIL_INLINE SUTIL_HOSTDEVICE float mod(float x, float y)
{ return x - y * floorf(x / y); }

SUTIL_INLINE SUTIL_HOSTDEVICE float3 mod(const float3 &v, float a)
{ return make_float3(mod(v.x, a), mod(v.y, a), mod(v.z, a)); }

SUTIL_INLINE SUTIL_HOSTDEVICE float fclampf(float x, float a, float b)
{ return fmaxf(a, fminf(x, b)); }

SUTIL_INLINE SUTIL_HOSTDEVICE int clampf(int x, int a, int b)
{ return a >= (b > x ? x : b) ? a : (b > x ? x : b); }

SUTIL_INLINE SUTIL_HOSTDEVICE float fmaxf4(float a, const float3 &bcd)
{ return fmaxf(fmaxf(a, bcd.x), fmaxf(bcd.y, bcd.z)); }

SUTIL_INLINE SUTIL_HOSTDEVICE float fminf4(float a, const float3 &bcd)
{ return fminf(fminf(a, bcd.x), fminf(bcd.y, bcd.z)); }