#ifndef __ZONOOPT_CONZONO_HPP__
#define __ZONOOPT_CONZONO_HPP__

#include "AbstractZono.hpp"
#include <exception>

namespace ZonoOpt
{

template<typename float_type>
class ConZono : public AbstractZono<float_type>
{
    public:

        // constructor
        ConZono() = default;

        ConZono(const Eigen::SparseMatrix<float_type>& G, const Eigen::Vector<float_type, -1>& c,
            const Eigen::SparseMatrix<float_type>& A, const Eigen::Vector<float_type, -1>& b, 
            bool zero_one_form=false)
        {
            set(G, c, A, b, zero_one_form);
        }

        // set method
        void set(const Eigen::SparseMatrix<float_type>& G, const Eigen::Vector<float_type, -1>& c,
            const Eigen::SparseMatrix<float_type>& A, const Eigen::Vector<float_type, -1>& b, 
            bool zero_one_form=false)
        {
            // check dimensions
            if (G.rows() != c.size() || A.rows() != b.size() || G.cols() != A.cols())
            {
            throw std::invalid_argument("ConZono: inconsistent dimensions.");
            }

            // conzono parameters
            this->G = G;
            this->A = A;
            this->c = c;
            this->b = b;
            this->nG = G.cols();
            this->nC = A.rows();
            this->n = G.rows();
            this->zero_one_form = zero_one_form;

            // abstract zono parameters
            this->nGc = this->nG;
            this->nGb = 0;
            this->Gc = this->G;
            this->Gb.resize(this->n, 0);
            this->Ac = this->A;
            this->Ab.resize(0, 0);
        }

        // get methods
        int get_n() const { return this->n; }
        int get_nG() const { return this->nG; }
        int get_nC() const { return this->nC; }
        Eigen::SparseMatrix<float_type> get_G() const { return this->G; }
        Eigen::SparseMatrix<float_type> get_A() const { return this->A; }
        Eigen::Vector<float_type, -1> get_c() const { return this->c; }
        Eigen::Vector<float_type, -1> get_b() const { return this->b; }
        bool is_0_1_form() const { return this->zero_one_form; }

        // generator conversion between [-1,1] and [0,1]
        void convert_form()
        {
            Eigen::Vector<float_type, -1> c, b;
            Eigen::SparseMatrix<float_type> G, A;

            if (!this->zero_one_form) // convert to [0,1] generators
            {
                c = this->c - this->G*Eigen::Vector<float_type, -1>::Ones(this->nG);
                b = this->b + this->A*Eigen::Vector<float_type, -1>::Ones(this->nG);
                G = 2*this->G;
                A = 2*this->A;

                set(G, c, A, b, true);
            }
            else // convert to [-1,1] generators
            {
                c = this->c + 0.5*this->G*Eigen::Vector<float_type, -1>::Ones(this->nG);
                b = this->b - 0.5*this->A*Eigen::Vector<float_type, -1>::Ones(this->nG);
                G = 0.5*this->G;
                A = 0.5*this->A;

                set(G, c, A, b, false);
            }
        }

        // display methods
        std::string print() const
        {
            std::stringstream ss;
            ss << "ConZono: " << std::endl;
            ss << "n: " << this->n << std::endl;
            ss << "nG: " << this->nG << std::endl;
            ss << "nC: " << this->nC << std::endl;
            ss << "G: " << Eigen::Matrix<float_type, -1, -1>(this->G) << std::endl;
            ss << "c: " << this->c << std::endl;
            ss << "A: " << Eigen::Matrix<float_type, -1, -1>(this->A) << std::endl;
            ss << "b: " << this->b << std::endl;
            ss << "zero_one_form: " << this->zero_one_form << std::endl;
            return ss.str();
        }

        // optimization

        // optimize over
        Eigen::Vector<float_type, -1> optimize_over( 
            const Eigen::SparseMatrix<float_type> &P, const Eigen::Vector<float_type, -1> &q, float_type c=0,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::optimize_over_admm(P, q, c, settings, solution);
        }

        // project point
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
            return AbstractZono<float_type>::is_empty_admm(settings, solution);
        }

        // support
        float_type support(const Eigen::Vector<float_type, -1>& d, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::support_admm(d, settings, solution);
        }

        // contains point
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
template <typename float_type>
ConZono<float_type> vrep_2_conzono(const Eigen::Matrix<float_type, -1, -1> &Vpoly)
{
    // dimensions
    int n_dims = Vpoly.cols();
    int n_verts = Vpoly.rows();

    // make generators
    Eigen::SparseMatrix<float_type> G = Vpoly.transpose().sparseView();
    Eigen::Vector<float_type, -1> c (n_dims);
    c.setZero();

    // make constraints
    std::vector<Eigen::Triplet<float_type>> tripvec;
    Eigen::SparseMatrix<float_type> A (1, n_verts);
    for (int i=0; i<n_verts; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(0, i, 1));
    }
    A.setFromTriplets(tripvec.begin(), tripvec.end());

    Eigen::Vector<float_type, -1> b (1);
    b(0) = 1;

    // return conzono
    return ConZono<float_type>(G, c, A, b, true);
}

} // namespace ZonoOpt


#endif