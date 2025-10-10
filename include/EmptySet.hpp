#ifndef ZONOOPT_EMPTYSET_HPP_
#define ZONOOPT_EMPTYSET_HPP_

/**
 * @file EmptySet.hpp
 * @author Josh Robbins (jrobbins@psu.edu)
 * @brief Empty Set class for ZonoOpt library.
 * @version 1.0
 * @date 2025-09-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ConZono.hpp"
#include "Zono.hpp"

namespace ZonoOpt {

/**
 * @brief Empty Set class.
 *
 * Used to facilitate set operations with trivial solutions when one of the sets is an empty set.
 */
class EmptySet final : public ConZono
{
public:

    /**
     * @brief Default constructor for EmptySet class
     *
     */
    EmptySet() = default;

    /**
     * @brief EmptySet constructor
     *
     * @param n dimension
     */
    explicit EmptySet(const int n)
    {
        this->n = n;

        // hybzono parameters
        this->c.resize(this->n);
        this->c.setConstant(std::numeric_limits<zono_float>::quiet_NaN());
        this->G.resize(this->n,0);
        this->nG = 0;
        this->nGc = this->nG;
        this->nGb = 0;
        this->nC = 0;
        this->Gc = this->G;
        this->Gb.resize(this->n, 0);
        this->A.resize(0, this->nG);
        this->Ac = this->A;
        this->Ab.resize(0, 0);
        this->b.resize(0);
        this->zero_one_form = false;
    }

    HybZono* clone() const override
    {
        return new EmptySet(*this);
    }

    std::string print() const override
    {
        std::stringstream ss;
        ss << "EmptySet: " << std::endl;
        ss << "  n: " << this->n;
        return ss.str();
    }

    void constraint_reduction() override { /* do nothing */ }

    std::unique_ptr<Zono> to_zono_approx() const override { throw std::runtime_error("to_zono_approx: EmptySet"); }

protected:
    Eigen::Vector<zono_float, -1> do_optimize_over(
            const Eigen::SparseMatrix<zono_float>&, const Eigen::Vector<zono_float, -1>&, zono_float,
            const OptSettings&, OptSolution* solution) const override
    {
        if (solution)
        {
            solution->infeasible = true;
        }
        return Eigen::Vector<zono_float, -1>::Constant(this->n, std::numeric_limits<zono_float>::quiet_NaN());
    }

    Eigen::Vector<zono_float, -1> do_project_point(const Eigen::Vector<zono_float, -1>&, const OptSettings&, OptSolution* solution) const override
    {
        if (solution)
        {
            solution->infeasible = true;
        }
        return Eigen::Vector<zono_float, -1>::Constant(this->n, std::numeric_limits<zono_float>::quiet_NaN());
    }

    zono_float do_support(const Eigen::Vector<zono_float, -1>&, const OptSettings&, OptSolution* solution) override
    {
        if (solution)
        {
            solution->infeasible = true;
        }
        return std::numeric_limits<zono_float>::quiet_NaN();
    }

    bool do_contains_point(const Eigen::Vector<zono_float, -1>&, const OptSettings&, OptSolution*) const override
    {
        return false;
    }

    Box do_bounding_box(const OptSettings&, OptSolution*) override
    {
        const Eigen::Vector<zono_float, -1> x_l = Eigen::Vector<zono_float, -1>::Constant(this->n, std::numeric_limits<zono_float>::infinity());
        const Eigen::Vector<zono_float, -1> x_u = -Eigen::Vector<zono_float, -1>::Constant(this->n, std::numeric_limits<zono_float>::infinity());
        return {x_l, x_u};
    }

    bool do_is_empty(const OptSettings&, OptSolution*) const override
    {
        return true;
    }

    std::unique_ptr<HybZono> do_complement(zono_float delta_m, bool, const OptSettings&, OptSolution*, int, int) override
    {
        const zono_float m = delta_m + 1; // box width
        const Eigen::Vector<zono_float, -1> x_l = -Eigen::Vector<zono_float, -1>::Constant(this->n, m);
        const Eigen::Vector<zono_float, -1> x_u = Eigen::Vector<zono_float, -1>::Constant(this->n, m);
        const Box box(x_l, x_u);
        return interval_2_zono(box);
    }
};

}

#endif