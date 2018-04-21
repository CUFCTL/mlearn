/**
 * @file cuda/device.cpp
 *
 * Implementation of the CUDA device type.
 */
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
	CHECK_CUBLAS(cublasCreate(&_cublas_handle));
	CHECK_CUSOLVER(cusolverDnCreate(&_cusolver_handle));
}



Device::~Device()
{
	CHECK_CUBLAS(cublasDestroy(_cublas_handle));
	CHECK_CUSOLVER(cusolverDnDestroy(_cusolver_handle));
}



}
