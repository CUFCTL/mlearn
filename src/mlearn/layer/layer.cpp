/**
 * @file layer/layer.cpp
 *
 * Implementation of the abstract layer type.
 */
#include "mlearn/layer/layer.h"



namespace mlearn {



IODevice& operator<<(IODevice& file, const Layer& layer)
{
	layer.save(file);
	return file;
}



IODevice& operator>>(IODevice& file, Layer& layer)
{
	layer.load(file);
	return file;
}



}
