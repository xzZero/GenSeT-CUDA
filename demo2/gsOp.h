#include <cufft.h>
void Iop_gs(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);
void Bhop_gs(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz);
void Ahop(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);
void AopCompl(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz);
void AopReal(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz);
void AopReal_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata_1, cufftDoubleComplex *outdata, double *S, long nnz);
void AopCompl_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, cufftDoubleComplex *outdata_1, double *S, long nnz);
void Ahop_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, cufftDoubleComplex *outdata_1, double *S, long nnz);
void Bhop_gs_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata_1, cufftDoubleComplex *outdata, double *S, long nnz);
void Iop_gs_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, cufftDoubleComplex *outdata_1, double *S, long nnz);
void AopReal(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize);
void AopCompl(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize);
void Ahop(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize);
void Bhop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize);
void Iop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize);