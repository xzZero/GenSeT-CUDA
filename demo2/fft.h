#pragma once
#include <cufft.h>
#include <cublas_v2.h>

void cuFFTR2C(cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuFFTC2C(cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuIFFTC2C(cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuFFTR2C_(cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuFFTC2C_(cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuIFFTC2C_(cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuFFTR2C(cudaStream_t stream, cufftHandle plan, cufftDoubleReal *indata, cuDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuFFTC2C(cudaStream_t stream, cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);
void cuIFFTC2C(cudaStream_t stream, cufftHandle plan, cublasHandle_t handle, cufftDoubleComplex *outdata, int Nslice, int My, int Mx);