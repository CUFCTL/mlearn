/**
 * @file feature/identity.cpp
 *
 * Implementation of identity feature layer.
 */
#include "feature/identity.h"
#include "util/logger.h"

namespace ML {

/**
 * Project an input matrix into the feature space
 * of the identity layer.
 *
 * @param X
 */
Matrix IdentityLayer::project(const Matrix& X)
{
	return X;
}

/**
 * Print information about an identity layer.
 */
void IdentityLayer::print()
{
	log(LL_VERBOSE, "Identity");
}

}
