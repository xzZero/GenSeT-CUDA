#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FileOpen.h"
#include "fft.h"
#include "gsOp.h"
#include<iostream>
#include "TimerCuda.h"
#include <time.h>
#include <cublas_v2.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include<Windows.h>
#include<fstream>

#define NN Nslice*My*Mx
long nnz = 220992;
int Nslice = 112;
int My = 128;
int Mx = 128;
float iternumCG = 250;
double tol = 1e-20;

void checkDeviceProperty(cudaDeviceProp deviceProp);
void DisplayHeader();
void saveBitmap(const char *p_output, int p_width, int p_height, unsigned char *Image);

void conjugateGradient(cufftHandle plan, cufftDoubleComplex *x_d, cufftDoubleComplex *r_d, cufftDoubleComplex *d_d, cufftDoubleComplex *u_d,
	cufftDoubleComplex *bestx_d, cufftDoubleComplex *u1_d, cufftDoubleComplex *outdata_, double *S, cufftDoubleReal *ref,
	int *iter, cufftDoubleComplex *bestx, cufftDoubleComplex *y);

void tv_nesterov_gear(cufftHandle plan, cublasHandle_t handle, double a, double mu, cufftDoubleComplex *b, double delta, cufftDoubleComplex *x0, int maxit, double norm_D, double norm_A, double lambda, int Nslice, int My, int Mx,
	cufftDoubleComplex *outdata_, cufftDoubleComplex *Dx, cufftDoubleComplex *tmp, cufftDoubleComplex *Ahb, cufftDoubleComplex *u_mu, cufftDoubleComplex* outputn, double *norm_u, double *S, long nnz);

void tv_nesterov_with_continuation_gear(cufftHandle plan, cublasHandle_t handle, double a, cufftDoubleComplex *b, double delta, double gamma, double mu_f, cufftDoubleComplex *x0, int maxit, double norm_D, double norm_A, double lambda, int Nslice, int My, int Mx,
	cufftDoubleComplex *outdata_, cufftDoubleComplex *Dx, cufftDoubleComplex *tmp, cufftDoubleComplex *tmp1, cufftDoubleComplex *Ahb, cufftDoubleComplex *u_mu, cufftDoubleComplex* outputn, cufftDoubleComplex* output, double *norm_u, double *S, long nnz);

void conjugateGradient_(cufftHandle plan, cufftDoubleComplex *x_d, cufftDoubleComplex *r_d, cufftDoubleComplex *d_d, cufftDoubleComplex *u_d,
	cufftDoubleComplex *bestx_d, cufftDoubleComplex *u1_d, cufftDoubleComplex *outdata_, double *S, cufftDoubleReal *ref,
	int *iter, cufftDoubleComplex *bestx, cufftDoubleComplex *y);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
//CHECK CUBLAS ERROR
#define cublasER(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cublasStatus_t code, char *file, int line, bool abort = true)
{
	if (code != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "cublasER: %s %s %d\n", code, file, line);
		if (abort) exit(code);
	}
}

__global__ void doubleMatrix(double *x, double *out, int Nslice, int My, int Mx, int residual) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 2);
	if (idx <= Nslice*My*Mx - residual - 1) {
		out[idx] = x[idx]*x[idx];
	}
}

__global__ void Sminus1(cufftDoubleReal *S) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	S[idx] = S[idx] - 1;
}

__global__ void abs_matrix(cufftDoubleComplex *x, double *y, int Nslice, int My, int Mx, int residual) {//check----------------------------

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 2);
	if (idx <= Nslice*My*Mx - residual - 1) {
		y[idx] = sqrt(x[idx].x*x[idx].x + x[idx].y*x[idx].y);
	}
}
__global__ void Checknorm(double *norm_u, double *out, double mu) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

	out[idx] = ((norm_u[idx] * norm_u[idx]) / (2 * mu)) * (norm_u[idx] < mu);
	if (out[idx] == 0) {
		out[idx] = (norm_u[idx] - mu / 2);
	}

}

__global__ void Dmultiply(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {   //check------------------
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	//idx = max(idx, 0);
	//idx = min(idx, Nslice*My*Mx - 3);
	if (idx <= Nslice*My*Mx - 2) {

		output[idx].x = x[idx].x - x[idx + 1].x;
		output[idx].y = x[idx].y - x[idx + 1].y;
	}
	// copy final var in x to output since D = 1
}
__global__ void DmultiplyAbs(cufftDoubleComplex *x, double *output, int Nslice, int My, int Mx) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	idx = max(idx, (long)0);
	idx = min(idx, (long)(Nslice*My*Mx - 2));

	output[idx] = (double)sqrt((x[idx].x - x[idx + 1].x)*(x[idx].x - x[idx + 1].x) + (x[idx].y - x[idx + 1].y)*(x[idx].y - x[idx + 1].y));

	// copy final var in x to output since D = 1
}
__global__ void DtransMultiply(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {
	long idx = blockIdx.x*blockDim.x + threadIdx.x;

	idx = max(idx, (long)1);
	idx = min(idx, (long)(Nslice*My*Mx - 2));

	output[idx].x = -x[idx - 1].x + x[idx].x;
	output[idx].y = -x[idx - 1].y + x[idx].y;
	// copy final var in x to output since D = 1
}
__global__ void muDivide(cufftDoubleComplex *x, cufftDoubleComplex *y, double mu) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	y[idx].x = x[idx].x / mu;
	y[idx].y = x[idx].y / mu;
}
__global__ void checkMu(double *norm_u, double mu, cufftDoubleComplex *u_mu, int Nslice, int My, int Mx) { //u_mu is Dx now, check-------------------

	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	idx = max(idx, (long)0);
	idx = min(idx, (long)(Nslice*My*Mx - 2));
	if (norm_u[idx] >= mu) {
		u_mu[idx].x = u_mu[idx].x * mu / norm_u[idx];
		u_mu[idx].y = u_mu[idx].y * mu / norm_u[idx];
	}

}

__global__ void mapping2GLubyte_range(double *in, int index_max, int index_min, double *out) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	out[idx] = (255 / (in[index_max - 1] - in[index_min - 1]))*(in[idx] - in[index_min - 1]);
}
__global__ void permute(double *in, double *out) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_in = (idx % 128) * 112 * 128 + ((idx / 128) % 128) * 112 + idx / (128 * 128);
	out[idx] = in[idx_in];
}
__global__ void cast(double *in, unsigned char *out) {
	int idx_out = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_in = blockIdx.x*blockDim.x + threadIdx.x + 128 * 128 * 10 + 1;
	out[idx_out] = static_cast<unsigned char>(in[idx_in]);
}
__global__ void expand(double *in, unsigned char *out) {
	int idx_in = blockIdx.x*blockDim.x + threadIdx.x;
	int Num = 128 * 128 * 16;
	int idx_out = idx_in % 128 + ((idx_in / 128) % 128) * (128 * 16) + ((idx_in % Num) / (128 * 128)) * 128 + (idx_in / Num) * Num;
	out[idx_out] = static_cast<unsigned char>(in[idx_in]);
}



void calDy(cufftDoubleComplex *x, cufftDoubleComplex *output, int Nslice, int My, int Mx) {  //another better solution ????
	cudaMemcpy(&output[0], &x[0], sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	cufftDoubleComplex t;
	t.x = 0.0f;
	t.y = 0.0f;
	cudaMemcpy(&t, &x[Nslice*My*Mx - 2], sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);

	t.x = t.x * (-1);
	t.y = t.y * (-1);

	cudaMemcpy(&output[Nslice*My*Mx - 1], &t, sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);


	//delete[] & t;
	//cannot use delete with static variable. Also, it is unecessary since t deleted auto when goes out of scope. 





	DtransMultiply << < Nslice*My*Mx / 128, 128 >> > (x, output, Nslice, My, Mx);
	//cudaDeviceSynchronize();

}
#define NUM_STREAMS 2
int main()
{
	DisplayHeader();
	size_t free, total;
	cudaMemGetInfo(&free, &total);

	std::cout << free / (1024 * 1024) << " MB" << std::endl;
	cufftHandle plan;
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}
	cufftResult result = cufftPlan3d(&plan, Mx, My, Nslice, CUFFT_Z2Z);
	if (result != CUFFT_SUCCESS) {   // the time to prepare this part is pretty long 
		fprintf(stderr, "CUFFT eror: PLan creation failed");
		std::cout << result << std::endl;
		return;
	}

	cudaMemGetInfo(&free, &total);

	std::cout << free / (1024 * 1024) << " MB" << std::endl;
	//StartCounter();
	// --- Creates CUDA streams
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

	cudaError_t status;
	//allocate host memory
	char link1[100];
	long nnz1[4] = { 4461440, 4641920, 4638720, 4459200 };
	double *ref_h, *rho_h;
	//double *S_h;
	//cufftDoubleComplex *data_h;


	//cufftComplex *data_h, *y_h, *rho_dft_h; 
	cufftDoubleComplex  **a_gs_h = new cufftDoubleComplex*[4];
	//status = cudaMallocHost((double**)&S_h, 3453*64 * sizeof(double));
	status = cudaMallocHost((double**)&ref_h, 128 * 128 * 112 * sizeof(double));
	status = cudaMallocHost((double**)&rho_h, 128 * 128 * 112 * sizeof(double));

	//status = cudaMallocHost((cufftDoubleComplex**)&a_gs_h, 3453 * 64 * sizeof(cufftDoubleComplex));
	//status = cudaMallocHost((cufftDoubleComplex**)&data_h, 3453 * 64 * sizeof(cufftDoubleComplex));

	double **S_h1 = new double *[4]; //Load 4 S_h first

	for (int ii = 0; ii < 4; ii++) {
		//gpuErrchk(cudaHostAlloc((double**)&S_h1[ii], nnz1[ii] * sizeof(double), cudaHostAllocPortable));
		cudaMallocHost((double**)&S_h1[ii], nnz1[ii] * sizeof(double));
		cudaMallocHost((cufftDoubleComplex**)&a_gs_h[ii], nnz1[ii] * sizeof(cufftDoubleComplex));
	}
	for (int jj = 1; jj < 5; jj++) {
		sprintf(link1, "D:/cuda demo/Frame_data_test/new/S%d.txt", jj);
		if (jj == 4) {
			FileOpen(link1, S_h1[0]);
			//std::cout << "4______4" << std::endl;
		}
		else {
			FileOpen(link1, S_h1[jj]);
			//std::cout << jj << std::endl;
		}
	}
	FileOpen("D:/cuda demo/Frame_data_test/new/Frame_1.txt", ref_h);
	std::cout << "-----------------" << std::endl;
	//allocate device memory


	double *ref_d, *rho_d;
	double *S_d;
	cufftDoubleComplex *rho_gs_d;
	cufftDoubleComplex **data_d = new cufftDoubleComplex*[4];
	cufftDoubleComplex **y_d = new cufftDoubleComplex*[4];
	cufftDoubleComplex **a_gs_d = new cufftDoubleComplex*[4];
	double **S_d1 = new double *[4];
	double *rho_abs, *rho_mapping, *rho_permute;
	unsigned char *checkImage_h, *checkImage_d;

	gpuErrchk(cudaMalloc(&rho_abs, Nslice*My*Mx * sizeof(double)));
	gpuErrchk(cudaMalloc(&rho_mapping, Nslice*My*Mx * sizeof(double)));
	gpuErrchk(cudaMalloc(&rho_permute, Nslice*My*Mx * sizeof(double)));
	gpuErrchk(cudaHostAlloc(&checkImage_h, Nslice*My*Mx * sizeof(unsigned char), cudaHostAllocWriteCombined));
	gpuErrchk(cudaMalloc(&checkImage_d, Nslice*My*Mx * sizeof(unsigned char)));

	for (int ii = 0; ii < 4; ii++) {
		gpuErrchk(cudaMalloc(&S_d1[ii], nnz1[ii] * sizeof(double)));
		gpuErrchk(cudaMalloc(&data_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&y_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
	}
	
	cudaMalloc(&ref_d, 128 * 128 * 112 * sizeof(double));
	cudaMalloc(&rho_d, 128 * 128 * 112 * sizeof(double));
	cudaMalloc(&rho_gs_d, 112 * 128 * 128 * sizeof(cufftDoubleComplex));

	gpuErrchk(cudaMemcpyAsync(S_d1[0], S_h1[0], nnz1[0] * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
	gpuErrchk(cudaMemcpyAsync(S_d1[1], S_h1[1], nnz1[1] * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
	gpuErrchk(cudaMemcpyAsync(S_d1[2], S_h1[2], nnz1[2] * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
	gpuErrchk(cudaMemcpyAsync(S_d1[3], S_h1[3], nnz1[3] * sizeof(double), cudaMemcpyHostToDevice, streams[1]));

	cudaMemGetInfo(&free, &total);
	for (int ii = 0; ii < 4; ii++) {

		//gpuErrchk(cudaHostAlloc((double**)&S_h[ii], nnz[ii] * sizeof(double), cudaHostAllocPortable));
		//std::cout << sizeof(S_h[ii]) << "\n";
		gpuErrchk(cudaFreeHost(S_h1[ii]));
	}

	std::cout << free / (1024 * 1024) << " MB" << std::endl;

	// transfer data from host to device. 
	//cudaMemcpyAsync(S_d, S_h, 56000 * sizeof(double), cudaMemcpyHostToDevice, streams[0]); 
	//cudaMemcpy(S_d, S_h, 3453 * 64 * sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(ref_d, ref_h, 128 * 128 * 112 * sizeof(double), cudaMemcpyHostToDevice);

	gpuErrchk(cudaFreeHost(ref_h));
	
	Sminus1 << <(nnz1[0] / 128), 128, 0, streams[0] >> > (S_d1[0]);
	Sminus1 << <(nnz1[1] / 128), 128, 0, streams[1] >> > (S_d1[1]);
	Sminus1 << <(nnz1[2] / 96), 96, 0, streams[0] >> > (S_d1[2]);
	Sminus1 << <(nnz1[3] / 128), 128, 0, streams[1] >> > (S_d1[3]);

	
//	Sminus1 << <(3453 * 64 / 32), 32 >> > (S_d);
	//cudaDeviceSynchronize();

	// begin for loop 
	int nframe = 0;
	int Nframe = 32;
	int n = 0;
	int iter = 0;

	// variable and device memory used uniquely for conjugate gradient

	cufftDoubleComplex *outdata_;
	cudaMalloc(&outdata_, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);
	AopReal(Nslice, My, Mx, plan, ref_d, outdata_, data_d[1], S_d1[1], nnz1[1]);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	float time;
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	std::cout << "----------               " << time << std::endl;

	cufftDoubleComplex **x_d = new cufftDoubleComplex *[4];
	cufftDoubleComplex **r_d = new cufftDoubleComplex *[4];
	cufftDoubleComplex **d_d = new cufftDoubleComplex *[4];
	cufftDoubleComplex **u_d = new cufftDoubleComplex *[4];
	cufftDoubleComplex **bestx_d = new cufftDoubleComplex *[4];
	cufftDoubleComplex **u1_d = new cufftDoubleComplex *[4];

	for (int ii = 0; ii < 4; ii++) {
		//x_d1[ii] = new cufftDoubleComplex[nnz[ii]];
		//x_d2[ii] = new cufftDoubleComplex[nnz[ii]];
		gpuErrchk(cudaMalloc(&x_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&r_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&d_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&u_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&bestx_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
		gpuErrchk(cudaMalloc(&u1_d[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
	}
	
	
	cudaMemGetInfo(&free, &total);

	std::cout << free/(1024*1024) << " MB"<< std::endl;

	// Prepare plan for CUFFT
	//cufftHandle plan;
	//if (cudaGetLastError() != cudaSuccess) {
	//	fprintf(stderr, "Cuda error: Failed to allocate\n");
	//	return;
	//}
	//cufftResult result = cufftPlan3d(&plan, Mx, My, Nslice, CUFFT_Z2Z);
	//if (result != CUFFT_SUCCESS) {   // the time to prepare this part is pretty long 
	//	fprintf(stderr, "CUFFT eror: PLan creation failed");
	//	std::cout << result << std::endl;
	//	return;
	//}

	std::cout << "-----------------" << std::endl;
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//prepare for nesterov variable
	cufftDoubleComplex *Dx, *tmp, *tmp1, *Ahb, *u_mu, *outputn, *output;
	double *norm_u;

	cufftDoubleComplex **temporary = new cufftDoubleComplex *[4];
	for (int ii = 0; ii < 4; ii++) {
		gpuErrchk(cudaMalloc(&temporary[ii], nnz1[ii] * sizeof(cufftDoubleComplex)));
	}
	cudaMemGetInfo(&free, &total);

	std::cout << free / (1024 * 1024) << " MB" << std::endl;
	//cudaMalloc(&temporary, nnz * sizeof(cufftDoubleComplex));
	cudaMalloc(&Dx, (Nslice*My*Mx - 1) * sizeof(cufftDoubleComplex));
	cudaMalloc(&tmp, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMalloc(&tmp1, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMalloc(&outputn, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMalloc(&output, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMalloc(&Ahb, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMalloc(&u_mu, (Nslice*My*Mx - 1) * sizeof(cufftDoubleComplex));
	cudaMalloc(&norm_u, (Nslice*My*Mx - 1) * sizeof(double));

	cudaMemset(tmp1, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	cublasHandle_t handle = 0;

	if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "CUBLAS	initialization failed!");
		return;
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	//Start of 131 frame 
	
	for (nframe = 1; nframe <= 32; nframe++) {
		int iii = nframe % 4;
		//StartCounter();

		std::cout << "Choosing k: " << iii << "     " << nnz1[iii] << std::endl;
		n = sprintf(link1, "D:/cuda demo/Frame_data_test/new/Frame_%.txt", nframe );

		FileOpen(link1, rho_h);

		cudaMemcpy(rho_d, rho_h, 128 * 128 * 112 * sizeof(double), cudaMemcpyHostToDevice);


		AopReal(Nslice, My, Mx, plan, rho_d, outdata_, data_d[iii], S_d1[iii], nnz1[iii]); //correct. 

		//AopReal_(Nslice, My, Mx, plan, rho_d, outdata_, outdata_1, outdata_2, S_d, nnz);
		Bhop_gs(Nslice, My, Mx, plan, ref_d, data_d[iii], outdata_, y_d[iii], S_d1[iii], nnz1[iii]);
		cudaDeviceSynchronize();

		/*cudaMemcpy(data_h, data_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		for (int j = 0; j <= 100; j++) {
			std::cout << data_h[j].x << " + " << data_h[j].y << std::endl;
		}

		std::cout << "============================================================================" << std::endl;
		cudaMemcpy(data_h, outdata_2, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		for (int j = 0; j <= 100; j++) {
			std::cout << data_h[j].x << " + " << data_h[j].y << std::endl;
		}

		cuDoubleComplex k = make_cuDoubleComplex(-1, 0);
		cublasZaxpy(handle, nnz, &k, data_d, 1, outdata_2, 1);

		std::cout << "============================================================================" << std::endl;
		cudaMemcpy(data_h, outdata_2, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		for (int j = 0; j <= 100; j++) {
			std::cout << data_h[j].x << " + " << data_h[j].y << std::endl;
		}
		double c;
		cublasDzasum(handle, nnz, outdata_2, 1, &c);
		std::cout << "result: " << c << std::endl;*/






		conjugateGradient(plan, x_d[iii], r_d[iii], d_d[iii], u_d[iii], bestx_d[iii], u1_d[iii], outdata_, S_d, ref_d, &iter, a_gs_h[iii], y_d[iii]);


		

		//std::cout << "This is the end. " << iter << "\n";

		Iop_gs(Nslice, My, Mx, plan, ref_d, bestx_d[iii], rho_gs_d, S_d1[iii], nnz1[iii]);

		/*cufftDoubleComplex *data_h;
		cudaMallocHost((cufftDoubleComplex**)&data_h, 64 * 64 * 112 * sizeof(cufftDoubleComplex));
		cudaMemcpy(data_h, rho_gs_d, 64 * 64 * 112 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		for (int ii = 0; ii < 100; ii++) {
			std::cout << data_h[ii].x << " + " << data_h[ii].y << " j " << std::endl;
		}*/
		double mu_f;
		Ahop(Nslice, My, Mx, plan, data_d[iii], outdata_, S_d1[iii], nnz1[iii]);
		cublasER(cublasDznrm2(handle, Nslice*My*Mx, outdata_, 1, &mu_f));
		mu_f = (double)1e-15 * 0.6 * mu_f;

	
		tv_nesterov_with_continuation_gear(plan, handle, 1, data_d[iii], 1e-04, 0.08, mu_f, rho_gs_d, 2, 2, 2, 100,
			Nslice, My, Mx, temporary[iii], Dx, tmp, tmp1, Ahb, u_mu, outputn, output, norm_u, S_d1[iii], nnz1[iii]);


		//std::cout << "This is the end. \n";

		//cufftDoubleComplex *checkOut;
		//cudaMallocHost(&checkOut, 320 * 320 * 112 * sizeof(cufftDoubleComplex));
		//cudaMemcpy(checkOut, output, 320 * 320 * 112 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		/*for (int j = 0; j <= 100; j++) {
			std::cout << checkOut[j].x << " + " << checkOut[j].y << std::endl;
		}*/

		
		int index_max, index_min;
		char link[300];
		abs_matrix << <112 * 128, 128 >> > (output, rho_abs, 112, 128, 128, 0);

		cudaMemcpy(ref_h, rho_abs, 128 * 128 * 112 * sizeof(double), cudaMemcpyDeviceToHost);

		cublasER(cublasIdamax(handle, Nslice*My*Mx, rho_abs, 1, &index_max));
		cublasER(cublasIdamin(handle, Nslice*My*Mx, rho_abs, 1, &index_min));

		mapping2GLubyte_range << <112 * 128, 128 >> > (rho_abs, index_max, index_min, rho_mapping);
		permute << <112 * 128, 128 >> > (rho_mapping, rho_permute);
		expand << <112 * 128, 128 >> > (rho_permute, checkImage_d);

		gpuErrchk(cudaMemcpy(checkImage_h, checkImage_d, 112 * 128 * 128 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	
		sprintf(link, "D:/test/Frame_%d.bmp", nframe);
		saveBitmap(link, 16* 128, 7* 128, checkImage_h);


		std::cout <<"-----------------" <<GetCounter() << std::endl;


		/*n = sprintf(link1, "D:/cuda demo/Frame_data_test/testdm.txt");

		double *out1;
		cudaMalloc(&out1, 112 * 320 * 320 * sizeof(double));
		FileOpen(link1, rho_h);
		cudaMemcpy(rho_d, rho_h, 320 * 320 * 112 * sizeof(double), cudaMemcpyHostToDevice);
		double c = -1;
		double c1, c2, c3;
		cublasER(cublasDaxpy(handle, 112 * 320 * 320, &c, rho_d, 1, rho_abs, 1));
		doubleMatrix << <112 * 320, 320 >> > (rho_abs, ref_d, 112, 320, 320, 0);
		doubleMatrix << <112 * 320, 320 >> > (rho_d, out1, 112, 320, 320, 0);
		cublasER(cublasDasum(handle, 112 * 320 * 320, ref_d, 1, &c));
		cublasER(cublasDasum(handle, 112 * 320 * 320, out1, 1, &c1));

		cublasER(cublasDnrm2(handle, 112 * 320 * 320, rho_abs, 1, &c2));
		cublasER(cublasDnrm2(handle, 112 * 320 * 320, rho_d, 1, &c3));
		std::cout << "--------Result---- 1 -----" << sqrt(c/c1) << std::endl;
		std::cout << "--------Result---- 2 -----" << c2 / c3 << std::endl;
		std::cout << "max  " << ref_h[index_max-1] << std::endl;
		std::cout << "min  " << ref_h[index_min-1] << std::endl;*/
		
	}


	cudaFree(ref_d);
	cudaFree(S_d);
	cudaFree(data_d);
	cudaFree(y_d);
	cudaFree(a_gs_d);

	cudaFree(rho_d);

	//Free device memory of conjugate gradient
	cudaFree(x_d);
	cudaFree(r_d);
	cudaFree(d_d);
	cudaFree(u_d);
	cudaFree(u1_d);
	cudaFree(bestx_d);
	cudaFree(outdata_);

	//Free host memory 
//	cudaFreeHost(S_h);
	cudaFreeHost(ref_h);
	cudaFreeHost(rho_h);
	cudaFreeHost(a_gs_h);
	//free(data_h); 
	//free(y_h); 

	//free(rho_dft_h); 
	cufftDestroy(plan);


	cudaDeviceReset();


    return 0;
}

void conjugateGradient(cufftHandle plan, cufftDoubleComplex * x_d, cufftDoubleComplex * r_d, cufftDoubleComplex * d_d, cufftDoubleComplex * u_d, cufftDoubleComplex * bestx_d, cufftDoubleComplex * u1_d, cufftDoubleComplex * outdata_, double * S, cufftDoubleReal * ref, int * iter, cufftDoubleComplex * bestx, cufftDoubleComplex * y)
{
	//Preparation for cublas
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	cufftDoubleComplex numerator, denom;


	numerator = make_cuDoubleComplex(0, 0);
	denom = make_cuDoubleComplex(0, 0);

	// set cuda Stream


	//cudaError_t result; 


	//Create streams and set device variable


	cudaMemset(x_d, 0, nnz * sizeof(cufftDoubleComplex));

	cudaMemcpy(d_d, y, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(r_d, y, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bestx_d, x_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);

	double delta0;
	cufftDoubleComplex delta_ = make_cuDoubleComplex(0, 0);

	//cublasStatus = cublasDznrm2(cublasHandle, size, d_d, 1, &delta0);
	cublasStatus = cublasZdotc(cublasHandle, nnz, r_d, 1, r_d, 1, &delta_);
	//cublasStatus = cublasDotcEx(cublasHandle, size, r_d, CUDA_C_32F, 1, r_d, CUDA_C_32F, 1, &delta_, CUDA_C_32F,CUDA_C_32F);
	delta0 = delta_.x;

	//std::cout << "delta_   " << delta_.x<<"+"<<delta_.y<<"j" << std::endl; 
	double delta = delta0;

	//std::cout << "delta: " << delta << std::endl;
	double bestres = 1;
	double gamma, res;
	double alpha = 0;
	int iter_cg;
	double delta_old;

	for (iter_cg = 2; iter_cg < iternumCG; iter_cg++) {
		//Bop, Bhop_gs input array, output array need to be on device memory
		//Bop_gs(d_d, u1_d);
		//Bhop_gs(u1_d, u_d);z

		Bhop_gs(Nslice, My, Mx, plan, ref, d_d, outdata_, u1_d, S, nnz);

		Bhop_gs(Nslice, My, Mx, plan, ref, u1_d, outdata_, u_d, S, nnz);   // u_d is wrong here. 




		if (cublasZdotc(cublasHandle, nnz, d_d, 1, r_d, 1, &numerator) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCdotc num failed" << std::endl;
		}



		//std::cout << "numerator: " << numerator.x<<"+"<<numerator.y<<"j" << std::endl; 



		if (cublasZdotc(cublasHandle, nnz, d_d, 1, u_d, 1, &denom) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCdotc deno failed" << std::endl;
		}


		alpha = (double)(numerator.x / denom.x);




		if (isnan(alpha)) {
			alpha = 0;
		}
		//std::cout << "alpha is: " << alpha << std::endl;


		if (cublasZaxpy(cublasHandle, nnz, &make_cuDoubleComplex(alpha, 0), d_d, 1, x_d, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for x = x+alpha*d failed" << std::endl;
		}


		if (cublasZaxpy(cublasHandle, nnz, &make_cuDoubleComplex(-alpha, 0), u_d, 1, r_d, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for r = r-alpha*u failed" << std::endl;
		}

		delta_old = delta;
		//cublasStatus = cublasDznrm2(cublasHandle, size, r_d, 1, &delta);
		cublasStatus = cublasZdotc(cublasHandle, nnz, r_d, 1, r_d, 1, &delta_);
		delta = delta_.x;


		//std::cout << "delta_ is" << delta_.x<<"+"<<delta_.y<<"j" << std::endl; 


		//std::cout << "value of delta is: " << delta << std::endl;
		gamma = (double)delta / delta_old;

		if (cublasZaxpy(cublasHandle, nnz, &make_cuDoubleComplex(1 / gamma, 0), r_d, 1, d_d, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "d = 1/gamma*r + d failed" << std::endl;
		}

		if (cublasZdscal(cublasHandle, nnz, &gamma, d_d, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "d = gamma*d failed" << std::endl;
		}

		res = (double)sqrt(delta / delta0);


		//std::cout << " res is: " << res << std::endl;

		if (res < bestres) {
			//if (cublasCcopy(cublasHandle, size, x_d, 1, bestx_d, 1) != CUBLAS_STATUS_SUCCESS) {
			//	std::cout << "res<bestres failed" << std::endl;
			//}
			cudaMemcpy(bestx_d, x_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
			bestres = res;
			//std::cout << "best res is: " << bestres << std::endl;
		}

		if (res < tol) {
			break;
		}
		//std::cout << " Loop " << iter_cg << " finished\n";

	}
	cudaMemcpy(bestx, bestx_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	*iter = iter_cg;
	//std::cout << "resolution is " << bestres << std::endl;

	cublasDestroy(cublasHandle);
}

void tv_nesterov_gear(cufftHandle plan, cublasHandle_t handle, double a, double mu, cufftDoubleComplex * b, double delta, cufftDoubleComplex * x0, int maxit, double norm_D, double norm_A, double lambda, int Nslice, int My, int Mx, cufftDoubleComplex * outdata_, cufftDoubleComplex * Dx, cufftDoubleComplex * tmp, cufftDoubleComplex * Ahb, cufftDoubleComplex * u_mu, cufftDoubleComplex * outputn, double * norm_u, double * S, long nnz)
{
	double L_mu = norm_D*norm_D / mu + 2 * lambda*norm_A*norm_A;
	double2 **x = new double2 *[maxit + 2];
	double2 **y = new double2 *[maxit + 1];
	double2 **nabla_f_mu = new double2 *[maxit + 1];
	double *f_mu = new double;
	double f_mu_bar = 0;
	double delta_f_mu = 0;
	double2 *x_temp;

	//Will be malloced outside then assign parameters inside to avoid leaking memory
	for (int ii = 0; ii < maxit + 1; ii++) {
		cudaMalloc((void**)&x[ii], Nslice*My*Mx * sizeof(double2));
		cudaMalloc((void**)&y[ii], Nslice*My*Mx * sizeof(double2));
		cudaMalloc((void**)&nabla_f_mu[ii], Nslice*My*Mx * sizeof(double2));
	}
	cudaMalloc((void**)&x_temp, Nslice*My*Mx * sizeof(double2));
	cudaMalloc((void**)&x[maxit + 1], Nslice*My*Mx * sizeof(double2));
	//cudaMalloc((void**)&f_mu, Nslice*My*Mx * sizeof(cufftComplex));
	f_mu = (double*)malloc((maxit + 1) * sizeof(double));
	//assign zero matrix to x, y, nabla_f_mu
	for (int ii = 0; ii < maxit + 1; ii++) {
		if (ii > 0) {
			cudaMemset(x[ii], 0, Nslice*My*Mx * sizeof(double2));
		}
		cudaMemset(y[ii], 0, Nslice*My*Mx * sizeof(double2));
		cudaMemset(nabla_f_mu[ii], 0, Nslice*My*Mx * sizeof(double2));
	}
	cudaMemset(x[maxit + 1], 0, Nslice*My*Mx * sizeof(double2));
	memset(f_mu, 0, (maxit + 1) * sizeof(double));

	cudaMemset(Dx, 0, (Nslice*My*Mx - 1) * sizeof(cufftDoubleComplex)); // Dx dim(Nslice - 1) due to D dim(N-1, N)
	cudaMemset(Ahb, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMemset(tmp, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));
	cudaMemset(outdata_, 0, nnz * sizeof(cufftDoubleComplex)); //need to change to nnz to suit Kmask

	cudaMemcpy((cufftDoubleComplex*)x[0], x0, Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);



	int k = 0;
	bool stopCriterion = false;

	Ahop(Nslice, My, Mx, plan, b, Ahb, S, nnz); //need to change dimension to suit data for kMask
	double aSqr = 1 / (a*a);
	if (cublasZdscal(handle, Nslice*My*Mx, &aSqr, Ahb, 1) != CUBLAS_STATUS_SUCCESS) {
		std::cout << "Ahb = Ah(b)/a^2" << std::endl;
	}

	cufftDoubleComplex *tmp2;
	cudaMalloc((void**)&tmp2, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	cufftDoubleComplex *b_test;
	cudaMalloc(&b_test, nnz * sizeof(cufftDoubleComplex));


	while (!stopCriterion) {
		// Step 1a -- Dim of Dx, u_mu, norm_u is Nslice*My*Mx - 1
		Dmultiply << <(Nslice*My*Mx / 128), 128 >> > (x[k], Dx, Nslice, My, Mx);
		cudaDeviceSynchronize();

		abs_matrix << <(Nslice*My*Mx / 128), 128 >> > (Dx, norm_u, Nslice, My, Mx, 1);
		cudaDeviceSynchronize();

		//u_mu is Dx by now since we do not need Dx
		double mu1 = (double)1 / mu;
		if (cublasZdscal(handle, Nslice*My*Mx - 1, &mu1, Dx, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "u_mu = Dx/mu" << std::endl;
		}

		checkMu << <(Nslice*My*Mx / 128), 128 >> > (norm_u, mu, Dx, Nslice, My, Mx);

		//Step 1b
		cudaMemcpy(x_temp, x[k], Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);


		AopCompl(Nslice, My, Mx, plan, x_temp, outdata_, S, nnz); // outdata_ dim must be nnz 
		cudaMemset(tmp, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));
		Ahop(Nslice, My, Mx, plan, outdata_, tmp, S, nnz);

		if (cublasZaxpy(handle, Nslice*My*Mx, &make_cuDoubleComplex(-1, 0), Ahb, 1, tmp, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for Ah(A(x(:,k))) - Ahb failed" << std::endl;
		}

		double lambda1 = 2 * lambda;
		if (cublasZdscal(handle, Nslice*My*Mx, &lambda1, tmp, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCsscal  2*lambda*(Ah(A(x(:,k)))-Ahb);" << std::endl;
		}

		calDy(Dx, nabla_f_mu[k], Nslice, My, Mx);

		if (cublasZaxpy(handle, Nslice*My*Mx, &make_cuDoubleComplex(1, 0), tmp, 1, nabla_f_mu[k], 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy  nabla_f_mu(:,k) = D'*u_mu + tmp failed" << std::endl;
		}

		//Step 2
		cudaMemcpy(y[k], x[k], Nslice*My*Mx * sizeof(double2), cudaMemcpyDeviceToDevice);
		
		

		if (cublasZaxpy(handle, Nslice*My*Mx, &make_cuDoubleComplex((double)-1 / L_mu, 0), nabla_f_mu[k], 1, y[k], 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for y(:,k) = x(:,k) - nabla_f_mu(:,k)/L_mu failed" << std::endl;
		}

		cufftDoubleComplex *y_check;
		cudaMallocHost(&y_check, Nslice*My*Mx * sizeof(double2));
		cudaMemcpy(y_check, y[k], Nslice*My*Mx * sizeof(double2), cudaMemcpyDeviceToHost);

		/*	for (int jj = 0; jj <= 100; jj++) {
				std::cout << " " << jj << " " << y_check[jj].x << "    " << y_check[jj].y << "j" << std::endl;
			}
		getchar();*/


		//Step 3 --- reuse tmp here
		cudaMemset(tmp, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));
		for (int i = 0; i <= k; i++) {
			//Assume cudaMemcpy override old data
			cudaMemcpy(tmp2, nabla_f_mu[i], Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
			double t = (double) 0.5 * (i + 1) / L_mu;
			if (cublasZdscal(handle, Nslice*My*Mx, &t, tmp2, 1) != CUBLAS_STATUS_SUCCESS) {
				std::cout << "cublasCsscal   tmp = nabla_f_mu(:,1:k)*(1/2*([0:k-1]+1))'/L_mu" << std::endl;
			}
			if (cublasZaxpy(handle, Nslice*My*Mx, &make_cuDoubleComplex(1, 0), tmp2, 1, tmp, 1) != CUBLAS_STATUS_SUCCESS) {
				std::cout << "cublasZaxpy for  tmp = nabla_f_mu(:,1:k)*(1/2*([0:k-1]+1))'/L_mu failed" << std::endl;
			}
		}

		//use tmp2 instead of z

		cudaMemcpy(tmp2, x[k], Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		if (cublasZaxpy(handle, Nslice*My*Mx, &make_cuDoubleComplex(-1, 0), tmp, 1, tmp2, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for   z = x(:,k) - tmp failed" << std::endl;
		}

		//Step 4
		//reuse tmp for  k/(k+2)*y(:,k)
		double beta = (double)(k + 1) / (k + 2 + 1); // since k = 0
		cudaMemcpy(tmp, y[k], Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		if (cublasZdscal(handle, Nslice*My*Mx, &beta, tmp, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCsscal   k/(k+2)*y(:,k)" << std::endl;
		}

			

		double alpha = (double)2 / (k + 2 + 1); // since k = 0
		if (cublasZdscal(handle, Nslice*My*Mx, &alpha, tmp2, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCsscal   2/(k+2)*z" << std::endl;
		}//////////////////////////

		if (cublasZaxpy(handle, Nslice*My*Mx, &make_cuDoubleComplex(1, 0), tmp, 1, tmp2, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for   z = x(:,k) - tmp failed" << std::endl;
		}

		//tmp2 is x(:, k + 1)

		cudaMemcpy(x[k + 1], tmp2, Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);

		//Step 6
		//Reuse Dx (u_mu by now) for Dy

		Dmultiply << <(Nslice*My*Mx / 128), 128 >> > (y[k], Dx, Nslice, My, Mx);
		cudaDeviceSynchronize();
		//Reuse norm_u also for "var"
		abs_matrix << <(Nslice*My*Mx / 128), 128 >> > (Dx, norm_u, Nslice, My, Mx, 1);
		cudaDeviceSynchronize();


		//---------------------------------------------------------------------------------------------------------

		double *norm_u1;
		cudaMalloc(&norm_u1, (Nslice*My*Mx - 1) * sizeof(double));

		Checknorm << <(Nslice*My*Mx / 128), 128 >> > (norm_u, norm_u1, mu);

		//reuse alpha and beta
		if (cublasDasum(handle, Nslice*My*Mx - 1, norm_u1, 1, &alpha) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasSasum for   sum(f) failed" << std::endl;
		}


		//A(y(:,k))


		cudaMemcpy(x_temp, y[k], Nslice*My*Mx * sizeof(double2), cudaMemcpyDeviceToDevice);

		AopCompl(Nslice, My, Mx, plan, x_temp, outdata_, S, nnz);



		//b-A(y(:,k))  --- reues outdata_ for b-A(y(:,k)) since we do not need it any more


		cudaMemcpy(b_test, b, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);

		if (cublasZaxpy(handle, nnz, &make_cuDoubleComplex(-1, 0), outdata_, 1, b_test, 1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for   b-A(y(:,k)) failed" << std::endl;
		}


		//norm(b - A(y(:, k))) -- beta here is L2 norm
		if (cublasDznrm2(handle, nnz, b_test, 1, &beta) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasScnrm2 for   norm(b - A(y(:, k))) failed" << std::endl;
		}
		//std::cout << "The value of norm is: " << beta << std::endl;
		//std::cout << "The value of sum is: " << alpha << std::endl;

		f_mu[k] = alpha + beta*beta;  //this part is considerably wrong. 
		//std::cout << "The value of f_mu is: " << f_mu[k] << std::endl;

		//std::cout << "VAlue of f_mu[k] is: " << f_mu[k] << std::endl; 


		if (k > 0) { // k start from 0
			alpha = 0;
			for (int ii = k - min(10, k); ii <= k - 1; ii++) {
				//reuse alpha for sum(f_mu)
				alpha += f_mu[ii];
			}
			f_mu_bar = (double)1 / min(10, k) * alpha;
			delta_f_mu = (double)abs(f_mu[k] - f_mu_bar) / f_mu_bar;

			if (delta_f_mu < delta) {
				stopCriterion = true;
			}

			if (f_mu[k] - f_mu[k - 1] > 0) {
				cudaMemcpy(y[k], y[k - 1], Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
				f_mu[k] = f_mu[k - 1];
				stopCriterion = true;
			}

		}

		if (k > maxit - 1) {
			stopCriterion = true;
		}

		k = k + 1;
		}
		//std::cout << "Value of k is" << k << std::endl;
		cudaMemcpy(outputn, y[k - 1], Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
}

void tv_nesterov_with_continuation_gear(cufftHandle plan, cublasHandle_t handle, double a, cufftDoubleComplex * b, double delta, double gamma, double mu_f, cufftDoubleComplex * x0, int maxit, double norm_D, double norm_A, double lambda, int Nslice, int My, int Mx, cufftDoubleComplex * outdata_, cufftDoubleComplex * Dx, cufftDoubleComplex * tmp, cufftDoubleComplex * tmp1, cufftDoubleComplex * Ahb, cufftDoubleComplex * u_mu, cufftDoubleComplex * outputn, cufftDoubleComplex * output, double * norm_u, double * S, long nnz)
{
	//we will init and malloc the parameters outside to eliminate the creation time
	//tmp for nesterov gear || tmp1 for tv_nesterov_with_continuation_gear
	//tmp1 dim is Nslice*My*Mx
	//x0 is for nesterov gear || x01 is for tv_nesterov_with_continuation_gear
	//outputn is for nesterov gear || output is for tv_nesterov_with_continuation_gear

	cudaMemset(tmp1, 0, Nslice*My*Mx * sizeof(cufftDoubleComplex));

	Ahop(Nslice, My, Mx, plan, b, tmp1, S, nnz);




	double mu_0 = 0;
	int idx;


	double *abs_tmp;
	cudaMalloc(&abs_tmp, Nslice*My*Mx * sizeof(double));


	abs_matrix << <(Nslice*My*Mx) / 128, 128 >> > (tmp1, abs_tmp, Nslice, My, Mx, 0);

	if (cublasIdamax(handle, Nslice*My*Mx, abs_tmp, 1, &idx) != CUBLAS_STATUS_SUCCESS) {
		std::cout << "failed  for finding maximum magnitude" << std::endl;
	}
	cudaMemcpy(&mu_0, &abs_tmp[idx - 1], sizeof(double), cudaMemcpyDeviceToHost);

	mu_0 = (double) mu_0*mu_0*0.9;

	//std::cout << mu_0 << std::endl;
	//getchar();
	//


	cudaMemcpy(output, x0, Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);



	double mu = mu_0;
	//std::cout << "mu " << mu << std::endl;
	// create parameters outside to avoid many malloc calls
	double *t1, *tn;
	double r1 = 0;
	double rn = 0;
	cudaMalloc((void**)&t1, (Nslice*My*Mx - 1) * sizeof(double));
	cudaMalloc((void**)&tn, (Nslice*My*Mx - 1) * sizeof(double));

	while (mu > mu_f) {
		tv_nesterov_gear(plan, handle, a, mu, b, delta, output, maxit, norm_D, norm_A, lambda,
			Nslice, My, Mx, outdata_, Dx, tmp, Ahb, u_mu, outputn, norm_u, S, nnz);

		//create 2 temporary variables 




		DmultiplyAbs << <(Nslice*My*Mx / 128), 128 >> > (outputn, tn, Nslice, My, Mx);
		cudaDeviceSynchronize();
		DmultiplyAbs << <(Nslice*My*Mx / 128), 128 >> > (output, t1, Nslice, My, Mx);
		cudaDeviceSynchronize();

		if (cublasDasum(handle, Nslice*My*Mx - 1, t1, 1, &r1) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasSasum for   sum(abs(D*output)) failed" << std::endl;
		}

		if (cublasDasum(handle, Nslice*My*Mx - 1, tn, 1, &rn) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasSasum for   sum(abs(D*outputn)) failed" << std::endl;
		}

		//std::cout << "___________________________________________________________" << std::endl;
		//std::cout << "rn = " << rn << std::endl;
		//std::cout << "r1 = " << r1 << std::endl;

		if (rn < r1) {
			cudaMemcpy(output, outputn, Nslice*My*Mx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
		}
		mu = mu * gamma;

	}

}

void conjugateGradient_(cufftHandle plan, cufftDoubleComplex * x_d, cufftDoubleComplex * r_d, cufftDoubleComplex * d_d, cufftDoubleComplex * u_d, cufftDoubleComplex * bestx_d, cufftDoubleComplex * u1_d, cufftDoubleComplex * outdata_, double * S, cufftDoubleReal * ref, int * iter, cufftDoubleComplex * bestx, cufftDoubleComplex * y)
{
	//Preparation for cublas
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	cufftDoubleComplex numerator, denom;


	numerator = make_cuDoubleComplex(0, 0);
	denom = make_cuDoubleComplex(0, 0);

	// set cuda Stream


	//cudaError_t result; 


	//Create streams and set device variable


	cudaMemset(x_d, 0, nnz * sizeof(cufftDoubleComplex));

	cudaMemcpy(d_d, y, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(r_d, y, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(bestx_d, x_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);

	double delta0;
	cufftDoubleComplex delta_ = make_cuDoubleComplex(0, 0);

	//cublasStatus = cublasDznrm2(cublasHandle, size, d_d, 1, &delta0);
	//cublasStatus = cublasZdotc(cublasHandle, nnz, r_d, 1, r_d, 1, &delta_);

	if (cublasDotcEx(cublasHandle, nnz, r_d, CUDA_C_64F, 1, r_d, CUDA_C_64F, 1, &delta_, CUDA_C_64F, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
		std::cout << "cublasCdotc num failed" << std::endl;
	}
	//cublasStatus = cublasDotcEx(cublasHandle, size, r_d, CUDA_C_32F, 1, r_d, CUDA_C_32F, 1, &delta_, CUDA_C_32F,CUDA_C_32F);
	delta0 = delta_.x;

	//std::cout << "delta_   " << delta_.x<<"+"<<delta_.y<<"j" << std::endl; 
	double delta = delta0;

	//std::cout << "delta: " << delta << std::endl;
	double bestres = 1;
	double gamma, res;
	double alpha = 0;
	int iter_cg;
	double delta_old;

	for (iter_cg = 2; iter_cg < iternumCG; iter_cg++) {
		//Bop, Bhop_gs input array, output array need to be on device memory
		//Bop_gs(d_d, u1_d);
		//Bhop_gs(u1_d, u_d);z

		Bhop_gs(Nslice, My, Mx, plan, ref, d_d, outdata_, u1_d, S, nnz);

		Bhop_gs(Nslice, My, Mx, plan, ref, u1_d, outdata_, u_d, S, nnz);   // u_d is wrong here. 


		if (cublasDotcEx(cublasHandle, nnz, d_d, CUDA_C_64F, 1, r_d, CUDA_C_64F, 1, &numerator, CUDA_C_64F, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCdotc num failed" << std::endl;
		}
		//std::cout << "numerator: " << numerator.x<<"+"<<numerator.y<<"j" << std::endl; 



		if (cublasDotcEx(cublasHandle, nnz, d_d, CUDA_C_64F, 1, u_d, CUDA_C_64F, 1, &denom, CUDA_C_64F, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasCdotc deno failed" << std::endl;
		}

		

		alpha = (double)(numerator.x / denom.x);




		if (isnan(alpha)) {
			alpha = 0;
		}
		//std::cout << "alpha is: " << alpha << std::endl;


		if (cublasAxpyEx(cublasHandle, nnz, &make_cuDoubleComplex(alpha, 0), CUDA_C_64F, d_d, CUDA_C_64F, 1, x_d, CUDA_C_64F, 1, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for x = x+alpha*d failed" << std::endl;
		}


		if (cublasAxpyEx(cublasHandle, nnz, &make_cuDoubleComplex(-alpha, 0), CUDA_C_64F, u_d, CUDA_C_64F, 1, r_d, CUDA_C_64F, 1, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "cublasZaxpy for r = r-alpha*u failed" << std::endl;
		}

		delta_old = delta;
		//cublasStatus = cublasDznrm2(cublasHandle, size, r_d, 1, &delta);
		cublasStatus = cublasDotcEx(cublasHandle, nnz, r_d, CUDA_C_64F, 1, r_d, CUDA_C_64F, 1, &delta_, CUDA_C_64F, CUDA_C_64F);
		delta = delta_.x;


		//std::cout << "delta_ is" << delta_.x<<"+"<<delta_.y<<"j" << std::endl; 


		//std::cout << "value of delta is: " << delta << std::endl;
		gamma = (double)delta / delta_old;

		if (cublasAxpyEx(cublasHandle, nnz, &make_cuDoubleComplex(1 / gamma, 0), CUDA_C_64F, r_d, CUDA_C_64F,  1, d_d, CUDA_C_64F, 1, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "d = 1/gamma*r + d failed" << std::endl;
		}

		if (cublasScalEx(cublasHandle, nnz, &make_cuDoubleComplex(gamma, 0), CUDA_C_64F, d_d, CUDA_C_64F, 1, CUDA_C_64F) != CUBLAS_STATUS_SUCCESS) {
			std::cout << "d = gamma*d failed" << std::endl;
		}

		res = (double)sqrt(delta / delta0);


		//std::cout << " res is: " << res << std::endl;

		if (res < bestres) {
			//if (cublasCcopy(cublasHandle, size, x_d, 1, bestx_d, 1) != CUBLAS_STATUS_SUCCESS) {
			//	std::cout << "res<bestres failed" << std::endl;
			//}
			cudaMemcpy(bestx_d, x_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
			bestres = res;
			//std::cout << "best res is: " << bestres << std::endl;
		}

		if (res < tol) {
			break;
		}
		//std::cout << " Loop " << iter_cg << " finished\n";

	}
	cudaMemcpy(bestx, bestx_d, nnz * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	*iter = iter_cg;
	//std::cout << "resolution is " << bestres << std::endl;
	
	cublasDestroy(cublasHandle);
}

void DisplayHeader()
{
	const int kb = 1024;
	const int mb = kb * kb;
	wcout << "NBody.GPU" << endl << "=========" << endl << endl;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;
	//wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

		wcout << "  Warp size:         " << props.warpSize << endl;
		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		checkDeviceProperty(props);
		wcout << endl;
	}
}


void checkDeviceProperty(cudaDeviceProp deviceProp)
{
	if ((deviceProp.concurrentKernels == 0)) //check concurrent kernel support
	{
		printf("> GPU does not support concurrent kernel execution\n");
		printf("  CUDA kernel runs will be serialized\n");
	}
	if (deviceProp.asyncEngineCount == 0) //check concurrent data transfer support
	{
		printf("GPU does not support concurrent Data transer and overlaping of kernel execution & data transfer\n");
		printf("Mem copy call will be blocking calls\n");
	}
}

void saveBitmap(const char *p_output, int p_width, int p_height, unsigned char *Image)
{
	BITMAPFILEHEADER bitmapFileHeader;
	memset(&bitmapFileHeader, 0xff, sizeof(BITMAPFILEHEADER));
	bitmapFileHeader.bfType = ('B' | 'M' << 8);
	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	bitmapFileHeader.bfSize = bitmapFileHeader.bfOffBits + p_width  * p_height; // multiply by 3 if you wanna switch to RGB

	BITMAPINFOHEADER bitmapInfoHeader;
	memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfoHeader.biWidth = p_width;
	bitmapInfoHeader.biHeight = -p_height;
	bitmapInfoHeader.biPlanes = 1;
	bitmapInfoHeader.biBitCount = 8; // this means grayscale, change to 24 if you wanna switch to RGB

	ofstream file(p_output, ios::binary);


	file.write(reinterpret_cast< char * >(&bitmapFileHeader), sizeof(bitmapFileHeader));
	file.write(reinterpret_cast< char * >(&bitmapInfoHeader), sizeof(bitmapInfoHeader));

	// bitmaps grayscale must have a table of colors... don't write this table if you want RGB
	unsigned char grayscale[4];

	for (int i(0); i < 256; ++i)
	{
		memset(grayscale, i, sizeof(grayscale));
		file.write(reinterpret_cast< char * >(grayscale), sizeof(grayscale));
	}

	// here we record the pixels... I created a gradient...
	// remember that the pixels ordem is from left to right, "bottom to top"...
	unsigned char pixel[1];
	for (int y(0); y < p_height; ++y)
	{
		for (int x(0); x < p_width; ++x) // + ( p_width % 4 ? ( 4 - p_width % 4 ) : 0 ) because BMP has a padding of 4 bytes per line
		{
			pixel[0] = Image[y*p_width + x];
			file.write(reinterpret_cast< char * >(pixel), sizeof(pixel));
		}
	}

	file.close();
}