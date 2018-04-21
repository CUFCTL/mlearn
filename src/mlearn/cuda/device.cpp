/**
 * @file cuda/device.cpp
 *
 * Implementation of the CUDA device type.
 */
#include <cassert>
#include "mlearn/cuda/device.h"



namespace ML {



std::unique_ptr<Device> Device::_instance;



void Device::initialize()
{
	if ( !_instance )
	{
		_instance.reset(new Device());
	}
}



Device * Device::instance()
{
	return _instance.get();
}



Device::Device()
{
	cublasStatus_t cublas_status = cublasCreate(&_cublas_handle);
	cusolverStatus_t cusolver_status = cusolverDnCreate(&_cusolver_handle);

	assert(cublas_status == CUBLAS_STATUS_SUCCESS && cusolver_status == CUSOLVER_STATUS_SUCCESS);
}



Device::~Device()
{
	cublasStatus_t cublas_status = cublasDestroy(_cublas_handle);
	cusolverStatus_t cusolver_status = cusolverDnDestroy(_cusolver_handle);

	assert(cublas_status == CUBLAS_STATUS_SUCCESS && cusolver_status == CUSOLVER_STATUS_SUCCESS);
}



}
