#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fft.h"
#include <stdlib.h>
#include <iostream>



__global__ void subsref(cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<nnz)
		outdata[idx] = indata[(long)S[idx]];
}
__global__ void subsasgn(cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	//*data must be memset
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<nnz)
		outdata[(long)S[idx]] = indata[idx];

}

__global__ void subsref_(cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	//if (idx<nnz/3)
		outdata[idx] = indata[(long)S[idx]];
		outdata[idx*2] = indata[(long)S[idx*2]];
		outdata[idx*3] = indata[(long)S[idx*3]];
}

__global__ void subsasgn_(cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	//*data must be memset
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	//if (idx<nnz)
		outdata[idx] = indata[(long)S[idx]];
		outdata[idx*2] = indata[(long)S[idx*2]];
		outdata[idx*3] = indata[(long)S[idx*3]];

}

__global__ void piecewiseMatMul(cufftDoubleReal *ref, cufftDoubleComplex *mat) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	mat[idx].x = ref[idx] * mat[idx].x;
	mat[idx].y = ref[idx] * mat[idx].y;
}

__global__ void piecewiseMatMul_(cufftDoubleReal *ref, cufftDoubleComplex *mat) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	mat[idx].x = ref[idx] * mat[idx].x;
	mat[idx].y = ref[idx] * mat[idx].y;

	mat[idx + 114688].x = ref[idx + 114688] * mat[idx + 114688].x;
	mat[idx + 114688].y = ref[idx + 114688] * mat[idx + 114688].y;

	mat[idx + 114688 * 2].x = ref[idx + 114688 * 2] * mat[idx + 114688 * 2].x;
	mat[idx + 114688 * 2].y = ref[idx + 114688 * 2] * mat[idx + 114688 * 2].y;

	mat[idx + 114688 * 3].x = ref[idx + 114688 * 3] * mat[idx + 114688 * 3].x;
	mat[idx + 114688 * 3].y = ref[idx + 114688 * 3] * mat[idx + 114688 * 3].y;
}

void AopReal(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz) {



	cuFFTR2C(plan, indata, outdata_, Nslice, My, Mx);

	if (nnz % 128 == 0) {
		subsref << <nnz / 128, 128 >> > (outdata_, outdata, S, nnz);
	}
	else {
		subsref << <nnz / 96, 96 >> > (outdata_, outdata, S, nnz);
	}
	//cudaDeviceSynchronize(); 
	//cudaFree(outdata_); 


}
void AopCompl(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {
	//AWARE:  INPUT OF AOPCOMPL IS CHANGED, SO DON'T USE INPUT AGAIN!!!! IF YOU WANT TO USE, COPY IT TO ANOTHER VARIABLE. 
	//cufftComplex *outdata_; 
	//cudaMalloc(&outdata_, Nslice*My*Mx * sizeof(cufftComplex));

	cuFFTC2C(plan, indata, Nslice, My, Mx);
	if (nnz % 128 == 0) {
		subsref << <nnz / 128, 128 >> > (indata, outdata, S, nnz);
	}
	else {
		subsref << <nnz / 96, 96 >> > (indata, outdata, S, nnz);
	}
	//cudaDeviceSynchronize();


	//cudaFree(outdata_); 

}
void Ahop(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {

	cudaMemset(outdata, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	if (nnz % 128 == 0) {
		subsasgn << <nnz / 128, 128 >> > (indata, outdata, S, nnz);
	}
	else {
		subsasgn << <nnz / 96, 96 >> > (indata, outdata, S, nnz);
	}

	//cudaDeviceSynchronize();

	cuIFFTC2C(plan, outdata, Nslice, My, Mx);



}
void Bhop_gs(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz) {



	//cudaMemset(outdata_, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));


	Ahop(Nslice, My, Mx, plan, indata, outdata_, S, nnz);

	piecewiseMatMul << <(Nslice*My*Mx / 256), 256  >> >(ref, outdata_);
	//cudaDeviceSynchronize();
	AopCompl(Nslice, My, Mx, plan, outdata_, outdata, S, nnz);
	//cudaFree(outdata_);

}

void Iop_gs(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz) {

	//cudaMemset(outdata, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	Ahop(Nslice, My, Mx, plan, indata, outdata, S, nnz);
	piecewiseMatMul << <(Nslice*My*Mx / 256), 256 >> >(ref, outdata);

}

void AopReal_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata_1, cufftDoubleComplex *outdata, double *S, long nnz) {



	cuFFTR2C_(plan, indata, outdata_1, outdata_, Nslice, My, Mx);


	subsref << <3453, 64 >> > (outdata_, outdata, S, nnz);
	//cudaDeviceSynchronize(); 
	//cudaFree(outdata_); 


}
void AopCompl_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, cufftDoubleComplex *outdata_1, double *S, long nnz) {
	//AWARE:  INPUT OF AOPCOMPL IS CHANGED, SO DON'T USE INPUT AGAIN!!!! IF YOU WANT TO USE, COPY IT TO ANOTHER VARIABLE. 
	//cufftComplex *outdata_; 
	//cudaMalloc(&outdata_, Nslice*My*Mx * sizeof(cufftComplex));

	cuFFTC2C_(plan, indata, outdata_1, Nslice, My, Mx);

	subsref << <3453, 64 >> > (outdata_1, outdata, S, nnz);
	//cudaDeviceSynchronize();


	//cudaFree(outdata_); 

}
void Ahop_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, cufftDoubleComplex *outdata_1, double *S, long nnz) {


	subsasgn << <3453, 64 >> > (indata, outdata_1, S, nnz);


	//cudaDeviceSynchronize();

	cuIFFTC2C_(plan, outdata_1, outdata, Nslice, My, Mx);



}
void Bhop_gs_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata_1, cufftDoubleComplex *outdata, double *S, long nnz) {



	cudaMemset(outdata_, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));


	Ahop_(Nslice, My, Mx, plan, indata, outdata_, outdata_1, S, nnz);

	piecewiseMatMul_ << <(Nslice*My*Mx / 256), 64 >> >(ref, outdata_);
	//cudaDeviceSynchronize();
	AopCompl_(Nslice, My, Mx, plan, outdata_, outdata, outdata_1, S, nnz);
	//cudaFree(outdata_);

}

void Iop_gs_(int Nslice, int My, int Mx, cufftHandle plan, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, cufftDoubleComplex *outdata_1, double *S, long nnz) {

	cudaMemset(outdata, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	Ahop_(Nslice, My, Mx, plan, indata, outdata, outdata_1, S, nnz);
	piecewiseMatMul_ << <(Nslice*My*Mx / 256), 64 >> >(ref, outdata);

}


void AopReal(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize) {



	cuFFTR2C(stream, plan, indata, outdata_, Nslice, My, Mx);


	subsref << <numBlock, blockSize, 0, stream >> > (outdata_, outdata, S, nnz);
	//cudaDeviceSynchronize(); 
	//cudaFree(outdata_); 


}

void AopCompl(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize) {

	//cufftComplex *outdata_; 
	//cudaMalloc(&outdata_, Nslice*My*Mx * sizeof(cufftComplex));

	cuFFTC2C(stream, plan, indata, Nslice, My, Mx);

	subsref << <numBlock, blockSize, 0, stream >> > (indata, outdata, S, nnz);
	//cudaDeviceSynchronize();


	//cudaFree(outdata_); 

}

void Ahop(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize) {


	subsasgn << <numBlock, blockSize, 0, stream >> > (indata, outdata, S, nnz);


	//cudaDeviceSynchronize();

	cuIFFTC2C(stream, plan, handle, outdata, Nslice, My, Mx);



}

void Bhop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize) {



	cudaMemset(outdata_, 0, Nslice*My*Mx * sizeof(cufftComplex));


	Ahop(Nslice, My, Mx, plan, handle, stream, indata, outdata_, S, nnz, numBlock, blockSize);

	piecewiseMatMul_ << <(Nslice*My*Mx / 256), 64, 0, stream >> >(ref, outdata_);
	//cudaDeviceSynchronize();
	AopCompl(Nslice, My, Mx, plan, handle, stream, outdata_, outdata, S, nnz, numBlock, blockSize);
	//cudaFree(outdata_);

}
void Iop_gs(int Nslice, int My, int Mx, cufftHandle plan, cublasHandle_t handle, cudaStream_t stream, cufftDoubleReal *ref, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, double *S, long nnz, long numBlock, long blockSize) {

	//cudaMemset(outdata, 0, Nslice*My*Mx * sizeof(cufftComplex));
	cudaMemsetAsync(outdata, 0, Nslice*My*Mx * sizeof(cufftComplex), stream);

	Ahop(Nslice, My, Mx, plan, handle, stream, indata, outdata, S, nnz, numBlock, blockSize);
	piecewiseMatMul_ << <(Nslice*My*Mx / 256), 64, 0, stream >> >(ref, outdata);

}