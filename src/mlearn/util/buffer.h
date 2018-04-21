/**
 * @file util/buffer.h
 *
 * Interface definitions for the buffer type.
 */
#ifndef BUFFER_H
#define BUFFER_H

#include <cuda_runtime.h>
#include "mlearn/cuda/device.h"



namespace ML {



template <class T>
class Buffer {
private:
	size_t _size {0};
	T *_host {nullptr};
	T *_dev {nullptr};

public:
	Buffer(size_t size, bool alloc_host=true);
	Buffer(const Buffer<T>& copy) = delete;
	Buffer(Buffer<T>&& move);
	Buffer() {}
	~Buffer();

	size_t size() const { return _size; }
	T * host_data() const { return _host; }
	T * device_data() const { return _dev; }

	void read();
	void write();

	Buffer& operator=(Buffer&& move);
};



template <class T>
Buffer<T>::Buffer(size_t size, bool alloc_host)
{
	_size = size;

	if ( Device::instance() )
	{
		if ( alloc_host )
		{
			CHECK_CUDA(cudaMallocHost(&_host, size * sizeof(T)));
		}

		CHECK_CUDA(cudaMalloc(&_dev, size * sizeof(T)));
	}
	else
	{
		_host = new T[size];
	}
}



template <class T>
Buffer<T>::Buffer(Buffer&& move)
	: Buffer()
{
	std::swap(_size, move._size);
	std::swap(_host, move._host);
	std::swap(_dev, move._dev);
}



template <class T>
Buffer<T>::~Buffer()
{
	if ( Device::instance() )
	{
		CHECK_CUDA(cudaFreeHost(_host));
		CHECK_CUDA(cudaFree(_dev));
	}
	else
	{
		delete[] _host;
	}
}



template <class T>
void Buffer<T>::read()
{
	if ( !Device::instance() ) {
		return;
	}

	CHECK_CUDA(cudaMemcpy(_host, _dev, _size * sizeof(T), cudaMemcpyDeviceToHost));
}



template <class T>
void Buffer<T>::write()
{
	if ( !Device::instance() ) {
		return;
	}

	CHECK_CUDA(cudaMemcpy(_dev, _host, _size * sizeof(T), cudaMemcpyHostToDevice));
}



template <class T>
Buffer<T>& Buffer<T>::operator=(Buffer<T>&& move)
{
	std::swap(_size, move._size);
	std::swap(_host, move._host);
	std::swap(_dev, move._dev);

	return *this;
}



}

#endif
