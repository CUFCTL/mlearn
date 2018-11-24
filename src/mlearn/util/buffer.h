/**
 * @file util/buffer.h
 *
 * Interface definitions for the buffer type.
 */
#ifndef MLEARN_UTIL_BUFFER_H
#define MLEARN_UTIL_BUFFER_H

#include <cuda_runtime.h>
#include "mlearn/cuda/device.h"



namespace mlearn {



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
	void read(size_t size, size_t offset=0);
	void write();
	void write(size_t size, size_t offset=0);

	Buffer<T>& operator=(Buffer<T>&& move);

	template <class U>
	friend void swap(Buffer<U>& lhs, Buffer<U>& rhs);
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
	swap(*this, move);
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
	read(_size);
}



template <class T>
void Buffer<T>::read(size_t size, size_t offset)
{
	if ( !Device::instance() ) {
		return;
	}

	CHECK_CUDA(cudaMemcpy(&_host[offset], &_dev[offset], size * sizeof(T), cudaMemcpyDeviceToHost));
}



template <class T>
void Buffer<T>::write()
{
	write(_size);
}



template <class T>
void Buffer<T>::write(size_t size, size_t offset)
{
	if ( !Device::instance() ) {
		return;
	}

	CHECK_CUDA(cudaMemcpy(&_dev[offset], &_host[offset], size * sizeof(T), cudaMemcpyHostToDevice));
}



template <class T>
Buffer<T>& Buffer<T>::operator=(Buffer<T>&& move)
{
	swap(*this, move);

	return *this;
}



template <class T>
void swap(Buffer<T>& lhs, Buffer<T>& rhs)
{
	std::swap(lhs._size, rhs._size);
	std::swap(lhs._host, rhs._host);
	std::swap(lhs._dev, rhs._dev);
}



}

#endif
