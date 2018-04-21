/**
 * @file cuda/device.h
 *
 * Interface definitions for the CUDA device type.
 */
#ifndef DEVICE_H
#define DEVICE_H

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <cublas_v2.h>
#include <cusolverDn.h>



namespace ML {



inline void CHECK_CUDA(cudaError_t err)
{
	if ( err != cudaSuccess ) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}



class Device {
private:
	static std::unique_ptr<Device> _instance;

	cublasHandle_t _cublas_handle;
	cusolverDnHandle_t _cusolver_handle;

public:
	static void initialize();
	static Device * instance();

	Device();
	~Device();

	cublasHandle_t cublas_handle() const { return _cublas_handle; }
	cusolverDnHandle_t cusolver_handle() const { return _cusolver_handle; }
};



}

#endif
