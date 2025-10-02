#ifndef ZONOOPT_POINT_HPP_
#define ZONOOPT_POINT_HPP_

/**
 * @file Point.hpp
 * @author Josh Robbins (jrobbins@psu.edu)
 * @brief Point class for ZonoOpt library.
 * @version 1.0
 * @date 2025-06-04
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "Zono.hpp"

namespace ZonoOpt
{

using namespace detail;

/**
 * @brief Point class.
 *
 * A point is defined entirely by the center vector c.
 */
class Point final : public Zono
{
    public:

        // constructor
        /**
         * @brief Default constructor for Point class
         *
         */
        Point() { sharp = true; }

        /**
         * @brief Point constructor
         *
         * @param c center
         */
        explicit Point(const Eigen::Vector<zono_float, -1>& c)
        {
            set(c);
            sharp = true;
        }

        // set method
        /**
         * @brief Reset point object with the given parameters.
         * 
         * @param c center
         */
        void set(const Eigen::Vector<zono_float, -1>& c);

        HybZono* clone() const override
        {
            return new Point(*this);
        }

        // display methods
        std::string print() const override;

        // do nothing methods
        void remove_redundancy(int interval_contractor) override { /* do nothing */ }
        void convert_form() override { /* do nothing */ }

    protected:

        Eigen::Vector<zono_float, -1> do_optimize_over(
            const Eigen::SparseMatrix<zono_float> &P, const Eigen::Vector<zono_float, -1> &q, zono_float c,
            const OptSettings &settings, OptSolution* solution) const override;

        Eigen::Vector<zono_float, -1> do_project_point(const Eigen::Vector<zono_float, -1>& x,
            const OptSettings &settings, OptSolution* solution) const override;

        zono_float do_support(const Eigen::Vector<zono_float, -1>& d, const OptSettings &settings,
            OptSolution* solution) override;

        bool do_contains_point(const Eigen::Vector<zono_float, -1>& x, const OptSettings &settings,
            OptSolution* solution) const override;

        Box do_bounding_box(const OptSettings &settings, OptSolution* solution) override;
};

// implementation
inline void Point::set(const Eigen::Vector<zono_float, -1>& c)
{
    // point parameters
    this->c = c;
    this->n = static_cast<int>(this->c.size());

    // hybzono parameters
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

inline std::string Point::print() const
{
    std::stringstream ss;
    ss << "Point: " << std::endl;
    ss << "n: " << this->n << std::endl;
    ss << "c: " << this->c;
    return ss.str();
}

inline Eigen::Vector<zono_float, -1> Point::do_optimize_over(
    const Eigen::SparseMatrix<zono_float> &P, const Eigen::Vector<zono_float, -1> &q, zono_float c,
    const OptSettings &settings, OptSolution* solution) const
{
    return this->c;
}

inline Eigen::Vector<zono_float, -1> Point::do_project_point(const Eigen::Vector<zono_float, -1>& x,
    const OptSettings &settings, OptSolution* solution) const
{
    // check dimensions
    if (this->n != x.size())
    {
        throw std::invalid_argument("Point projection: inconsistent dimensions.");
    }

    return this->c;
}

inline zono_float Point::do_support(const Eigen::Vector<zono_float, -1>& d,
    const OptSettings &settings, OptSolution* solution)
{
    // check dimensions
    if (this->n != d.size())
    {
        throw std::invalid_argument("Support: inconsistent dimensions.");
    }

    return this->c.dot(d);
}


inline bool Point::do_contains_point(const Eigen::Vector<zono_float, -1>& x,
    const OptSettings &settings, OptSolution* solution) const
{
    if (this->n != x.size())
        throw std::invalid_argument("Contains point: inconsistent dimensions");
    
    const zono_float dist = (x - this->c).norm();
    return dist < zono_eps;
}

inline Box Point::do_bounding_box(const OptSettings &settings, OptSolution* solution)
{
    return {this->c, this->c};
}

} // namespace ZonoOpt

#endif