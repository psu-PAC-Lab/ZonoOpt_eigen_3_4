#ifndef __ZONOOPT_POINT_HPP__
#define __ZONOOPT_POINT_HPP__

#include "AbstractZono.hpp"

namespace ZonoOpt
{

template<typename float_type>
class Point : public AbstractZono<float_type>
{
    public:

        // constructor
        Point() = default;

        Point(const Eigen::Vector<float_type, -1>& c)
        {
            set(c);
        }

        // set method
        void set(const Eigen::Vector<float_type, -1>& c)
        {
            // point parameters
            this->c = c;
            this->n = this->c.size();

            // abstract zono parameters
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

        // get methods
        int get_n() const { return this->n; }
        Eigen::Vector<float_type, -1> get_c() const { return this->c; }

        virtual void convert_form(){ /* do nothing */ }

        // display methods
        std::string print() const
        {
            std::stringstream ss;
            ss << "Point: " << std::endl;
            ss << "n: " << this->n << std::endl;
            ss << "c: " << this->c << std::endl;
            return ss.str();
        }

        // remove redundancy
        void remove_redundancy()
        {
            // do nothing
        }

        // optimization
        Eigen::Vector<float_type, -1> optimize_over( 
            const Eigen::SparseMatrix<float_type> &P, const Eigen::Vector<float_type, -1> &q, float_type c=0,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const
        {
            throw std::invalid_argument("Optimize over: cannot optimize over point.");
        }

        Eigen::Vector<float_type, -1> project_point(const Eigen::Vector<float_type, -1>& x, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const
        {
            // check dimensions
            if (this->n != x.size())
            {
                throw std::invalid_argument("Point projection: inconsistent dimensions.");
            }

            return this->c;
        }

        bool is_empty(const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            if (this->n == 0)
                return true;
            else
                return false;
        }

        float_type support(const Eigen::Vector<float_type, -1>& d, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            // check dimensions
            if (this->n != d.size())
            {
                throw std::invalid_argument("Support: inconsistent dimensions.");
            }

            return this->c.dot(d);
        }

        bool contains_point(const Eigen::Vector<float_type, -1>& x,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            if (this->n != x.size())
                throw std::invalid_argument("Contains point: inconsistent dimensions");
            
            for (int i=0; i<this->n; i++)
            {
                if (x(i) != this->c(i))
                    return false;
            }
            return true;
        }

        std::unique_ptr<AbstractZono<float_type>> bounding_box(
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return std::make_unique<Point<float_type>>(this->c);
        }

};

} // namespace ZonoOpt

#endif