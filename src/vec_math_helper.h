// Copyright (c) 2022-2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
