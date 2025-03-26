#ifndef __ZONOOPT_HPP__
#define __ZONOOPT_HPP__

#define EIGEN_MPL2_ONLY // Disable features licensed under LGPL

#include "Point.hpp"
#include "Zono.hpp"
#include "ConZono.hpp"
#include "HybZono.hpp"
#include "PolymorphicFunctions.hpp"
#include "ADMM.hpp"

namespace ZonoOpt
{
    typedef std::unique_ptr<AbstractZono<float>> ZonoPtrF;
    typedef std::unique_ptr<AbstractZono<double>> ZonoPtrD;
}

#endif