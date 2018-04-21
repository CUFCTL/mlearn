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
#include "mlearn/util/error.h"



namespace ML {



#define CHECK_CUDA(ret) \
	CHECK_ERROR(ret == cudaSuccess, #ret)



#define CHECK_CUBLAS(ret) \
	CHECK_ERROR(ret == CUBLAS_STATUS_SUCCESS, #ret)



#define CHECK_CUSOLVER(ret) \
	CHECK_ERROR(ret == CUSOLVER_STATUS_SUCCESS, #ret)



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
