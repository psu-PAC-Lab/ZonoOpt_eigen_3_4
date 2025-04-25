#ifndef __ZONOOPT_ZONO_HPP__
#define __ZONOOPT_ZONO_HPP__

#include "AbstractZono.hpp"
#include <exception>
#include <cmath>

namespace ZonoOpt
{

template<typename float_type>
class Zono : public AbstractZono<float_type>
{
    public:

        // constructor
        Zono() = default;

        Zono(const Eigen::SparseMatrix<float_type>& G, const Eigen::Vector<float_type, -1>& c,
            bool zero_one_form=false)
        {
            set(G, c, zero_one_form);
        }

        // set method
        void set(const Eigen::SparseMatrix<float_type>& G, const Eigen::Vector<float_type, -1>& c,
            bool zero_one_form=false)
        {
            // check dimensions
            if (G.rows() != c.size())
            {
                throw std::invalid_argument("Zono: inconsistent dimensions.");
            }

            // zonotope parameters
            this->G = G;
            this->c = c;
            this->nG = this->G.cols();
            this->n = this->G.rows();
            this->zero_one_form = zero_one_form;

            // abstract zono parameters
            this->nGc = this->nG;
            this->nGb = 0;
            this->nC = 0;
            this->Gc = this->G;
            this->Gb.resize(this->n, 0);
            this->A.resize(0, this->nG);
            this->Ac = this->A;
            this->Ab.resize(0, 0);
            this->b.resize(0);
        }

        // get methods
        int get_n() const { return this->n; }
        int get_nG() const { return this->nG; }
        Eigen::SparseMatrix<float_type> get_G() const { return this->G; }
        Eigen::Vector<float_type, -1> get_c() const { return this->c; }
        bool is_0_1_form() const { return this->zero_one_form; }

        // generator conversion between [-1,1] and [0,1]
        void convert_form()
        {
            Eigen::Vector<float_type, -1> c;
            Eigen::SparseMatrix<float_type> G;

            if (!this->zero_one_form) // convert to [0,1] generators
            {
                c = this->c - this->G*Eigen::Vector<float_type, -1>::Ones(this->nG);
                G = 2*this->G;

                set(G, c, true);
            }
            else // convert to [-1,1] generators
            {
                c = this->c + 0.5*this->G*Eigen::Vector<float_type, -1>::Ones(this->nG);
                G = 0.5*this->G;

                set(G, c, false);
            }
        }

        // display methods
        std::string print() const
        {
            std::stringstream ss;
            ss << "Zono: " << std::endl;
            ss << "n: " << this->n << std::endl;
            ss << "nG: " << this->nG << std::endl;
            ss << "G: " << Eigen::Matrix<float_type, -1, -1>(this->G) << std::endl;
            ss << "c: " << this->c << std::endl;
            ss << "zero_one_form: " << this->zero_one_form << std::endl;
            return ss.str();
        }

        // optimization
        Eigen::Vector<float_type, -1> optimize_over( 
            const Eigen::SparseMatrix<float_type> &P, const Eigen::Vector<float_type, -1> &q, float_type c=0,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::optimize_over_admm(P, q, c, settings, solution);
        }

        // project point onto set
        Eigen::Vector<float_type, -1> project_point(const Eigen::Vector<float_type, -1>& x, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::project_point_admm(x, settings, solution);
        }

        // is empty
        bool is_empty(const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            if (this->n == 0)
                return true;
            else
                return false;
        }

        // support
        float_type support(const Eigen::Vector<float_type, -1>& d, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::support_admm(d, settings, solution);
        }

        // point containment
        bool contains_point(const Eigen::Vector<float_type, -1>& x,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::contains_point_admm(x, settings, solution);
        }

        // bounding box
        std::unique_ptr<AbstractZono<float_type>> bounding_box(
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::bounding_box_admm(settings, solution);
        }

};

// setup functions
template<typename float_type>
std::unique_ptr<Zono<float_type>> make_regular_zono_2D(float_type radius, int n_sides, bool outer_approx=false, Eigen::Vector<float_type, 2>& c=Eigen::Vector<float_type, 2>::Zero())
{
    // check number of sides
    if (n_sides % 2 != 0 || n_sides < 4)
    {
        throw std::invalid_argument("make_regular_zono_2D: number of sides must be even and >= 4.");
    }

    // check radius
    if (radius <= 0)
    {
        throw std::invalid_argument("make_regular_zono_2D: radius must be positive.");
    }

    // problem parameters
    int n_gens = n_sides/2;
    const float_type pi = 3.14159265358979323846;
    float_type dphi = pi/n_gens;
    float_type R = outer_approx ? radius/std::cos(dphi/2) : radius;
    
    // generator matrix
    float_type phi = ((float_type) (n_gens/2))*dphi;
    float_type l_side = 2*R*std::sin(dphi/2);
    Eigen::Matrix<float_type, -1, -1> G(2, n_gens);
    for (int i = 0; i < n_gens; i++)
    {
        G(0, i) = l_side*std::cos(phi);
        G(1, i) = l_side*std::sin(phi);
        phi -= dphi;
    }

    // return zonotope
    return std::make_unique<Zono<float_type>>(0.5*G.sparseView(), c, false);
}

template <typename float_type>
std::unique_ptr<Zono<float_type>> interval_2_zono(const Eigen::Vector<float_type, -1>& a, const Eigen::Vector<float_type, -1>& b)
{
    // check dimensions
    if (a.size() != b.size())
    {
        throw std::invalid_argument("Interval to zonotope: inconsistent dimensions.");
    }

    // generator matrix
    std::vector<Eigen::Triplet<float_type>> triplets;
    Eigen::SparseMatrix<float_type> G (a.size(), a.size());
    for (int i=0; i<a.size(); i++)
    {
        triplets.push_back(Eigen::Triplet<float_type>(i, i, (b(i)-a(i))/2));
    }
    G.setFromSortedTriplets(triplets.begin(), triplets.end());

    // center
    Eigen::Vector<float_type, -1> c  = (a+b)/2;

    // return zonotope
    return std::make_unique<Zono<float_type>>(G, c, false);
}

} // namespace ZonoOpt

#endif