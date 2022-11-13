#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cufft.h>
#include<cublas_v2.h>
#include<iostream>

#define BATCH 1
__global__ void real2complex(cufftDoubleReal *in, cufftDoubleComplex *out) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	out[i] = make_cuDoubleComplex(in[i], 0);
}

void cuFFTR2C(cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {

	int nElem = Nslice*My*Mx;
	dim3 block(128);
	real2complex << < ((nElem + block.x - 1) / block.x), (block.x) >> > (indata, outdata);





	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}
}

void cuFFTC2C(cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {




	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}

}

void cuIFFTC2C(cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {   // because C-> C we only need 1 variable: outdata. 


	cublasHandle_t handle;
	cublasStatus_t status;
	double alpha = double(1) / (Nslice*My*Mx);



	status = cublasCreate(&handle);



	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



	if (cublasZdscal(handle, Nslice*My*Mx, &alpha, outdata, 1) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Cuda eror: Failed to synchronize ---- cuIFFTC2C\n");
		return;
	}

	cublasDestroy(handle);

}

void cuFFTR2C_(cufftHandle plan, cufftDoubleReal *indata, cufftDoubleComplex *outdata_, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {

	int nElem = Nslice*My*Mx;
	dim3 block(128);
	real2complex << < ((nElem + block.x - 1) / block.x), (block.x) >> > (indata, outdata_);





	if (cufftExecZ2Z(plan, outdata_, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}
}

void cuFFTC2C_(cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {




	if (cufftExecZ2Z(plan, indata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}

}

void cuIFFTC2C_(cufftHandle plan, cufftDoubleComplex *indata, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {   // because C-> C we only need 1 variable: outdata. 


	cublasHandle_t handle;
	cublasStatus_t status;
	double alpha = double(1) / (Nslice*My*Mx);



	status = cublasCreate(&handle);



	if (cufftExecZ2Z(plan, indata, outdata, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



	if (cublasZdscal(handle, Nslice*My*Mx, &alpha, outdata, 1) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Cuda eror: Failed to synchronize ---- cuIFFTC2C\n");
		return;
	}

	cublasDestroy(handle);

}

void cuFFTR2C(cudaStream_t stream, cufftHandle plan, cufftDoubleReal *indata, cuDoubleComplex *outdata, int Nslice, int My, int Mx) {

	int nElem = Nslice*My*Mx;
	dim3 block(64);
	real2complex << < (nElem / block.x), (block.x), 0, stream >> > (indata, outdata);





	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



}
void cuFFTC2C(cudaStream_t stream, cufftHandle plan, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {




	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}

}

void cuIFFTC2C(cudaStream_t stream, cufftHandle plan, cublasHandle_t handle, cufftDoubleComplex *outdata, int Nslice, int My, int Mx) {   // because C-> C we only need 1 variable: outdata. 


																																	//cublasHandle_t handle;
																																	//cublasStatus_t status;
	double alpha = float(1) / (Nslice*My*Mx);



	//status = cublasCreate(&handle);



	if (cufftExecZ2Z(plan, outdata, outdata, CUFFT_INVERSE) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecD2Z forward failed");
		return;
	}
	//if (cudaDeviceSynchronize() != cudaSuccess) {
	//	fprintf(stderr, "Cuda eror: Failed to synchronize\n");
	//	return;
	//}



	cublasZdscal(handle, Nslice*My*Mx, &alpha, outdata, 1);

}