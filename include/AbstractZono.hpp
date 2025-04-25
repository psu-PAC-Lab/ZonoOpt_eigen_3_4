#ifndef __ZONOOPT_ABSTRACTZONO_HPP__
#define __ZONOOPT_ABSTRACTZONO_HPP__

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <utility>
#include <memory>
#include "ADMM.hpp"

namespace ZonoOpt
{

// forward declarations
template <typename float_type>
class AbstractZono;

template <typename float_type>
class Point;

template <typename float_type>
class Zono;

template <typename float_type>
class ConZono;

template <typename float_type>
class HybZono;

template <typename float_type>
class AbstractZono
{
    public:

        virtual ~AbstractZono() = default; // abstract destructor
        virtual void convert_form() = 0; // abstract
        virtual std::string print() const = 0; // abstract

        template <typename T>
        friend std::ostream& operator<<(std::ostream &os, const AbstractZono<T>& Z);

        // set operation declarations
        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> affine_map(const AbstractZono<T>& Z,
            const Eigen::SparseMatrix<T>& R, const Eigen::Vector<T, -1>& s);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> project_onto_dims(const AbstractZono<T>& Z, std::vector<int>& dims);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> minkowski_sum(const AbstractZono<T>& Z1, AbstractZono<T>& Z2);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> pontry_diff(AbstractZono<T>& Z1, AbstractZono<T>& Z2);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> intersection(const AbstractZono<T>& Z1, AbstractZono<T>& Z2, 
            const Eigen::SparseMatrix<T>& R);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> intersection_over_dims(const AbstractZono<T>& Z1, AbstractZono<T>& Z2, 
            std::vector<int>& dims);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> halfspace_intersection(AbstractZono<T>& Z, const Eigen::SparseMatrix<T>& H, 
            const Eigen::Vector<T, -1>& f, const Eigen::SparseMatrix<T>& R);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> union_of_many(std::vector<AbstractZono<T>*>& Zs);

        template <typename T>
        friend std::unique_ptr<AbstractZono<T>> cartesian_product(const AbstractZono<T>& Z1, AbstractZono<T>& Z2);

        // setup methods
        template <typename T>
        friend HybZono<T> zono_union_2_hybzono(std::vector<AbstractZono<T>*>& Zs);

        // remove redundancy
        virtual void remove_redundancy()
        {
            // remove redundant constraints
            remove_redundant<float_type>(this->A, this->b);
            this->nC = this->A.rows();
            
            // identify unused generators
            std::vector<int> idx_to_remove = find_unused_generators(this->G, this->A);

            // remove generators
            if (idx_to_remove.size() != 0)
            {
                remove_generators(this->G, this->A, idx_to_remove);
            }

            // update equivalent matrices
            this->Gc = this->G;
            this->Ac = this->A;

            // update number of generators
            this->nG = this->G.cols();
            this->nGc = this->nG;
        }

        // optimization
        virtual Eigen::Vector<float_type, -1> optimize_over( 
            const Eigen::SparseMatrix<float_type> &P, const Eigen::Vector<float_type, -1> &q, float_type c=0,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const = 0;

        virtual Eigen::Vector<float_type, -1> project_point(const Eigen::Vector<float_type, -1>& x, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const = 0;

        virtual bool is_empty(const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* sol=nullptr) const = 0;

        virtual float_type support(const Eigen::Vector<float_type, -1>& d, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const = 0;

        virtual bool contains_point(const Eigen::Vector<float_type, -1>& x,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const = 0;

        virtual std::unique_ptr<AbstractZono<float_type>> bounding_box(
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const = 0;

        // type checking
        bool is_point() const
        {
            return dynamic_cast<const Point<float_type>*>(this) != nullptr;
        }

        bool is_zono() const
        {
            return dynamic_cast<const Zono<float_type>*>(this) != nullptr;
        }

        bool is_conzono() const
        {
            return dynamic_cast<const ConZono<float_type>*>(this) != nullptr;
        }

        bool is_hybzono() const
        {
            return dynamic_cast<const HybZono<float_type>*>(this) != nullptr;
        }

    protected:

        // fields
        Eigen::SparseMatrix<float_type> G = Eigen::SparseMatrix<float_type>(0, 0);
        Eigen::SparseMatrix<float_type> Gc = Eigen::SparseMatrix<float_type>(0, 0);
        Eigen::SparseMatrix<float_type> Gb = Eigen::SparseMatrix<float_type>(0, 0);
        Eigen::SparseMatrix<float_type> A = Eigen::SparseMatrix<float_type>(0, 0);
        Eigen::SparseMatrix<float_type> Ac = Eigen::SparseMatrix<float_type>(0, 0); 
        Eigen::SparseMatrix<float_type> Ab = Eigen::SparseMatrix<float_type>(0, 0);
        Eigen::Vector<float_type, -1> c = Eigen::Vector<float_type, -1>(0);
        Eigen::Vector<float_type, -1> b = Eigen::Vector<float_type, -1>(0);
        int n = 0;
        int nG = 0;
        int nGc = 0;
        int nGb = 0;
        int nC = 0;
        bool zero_one_form = false;

    
        // optimization
        Eigen::Vector<float_type, -1> optimize_over_admm( 
            const Eigen::SparseMatrix<float_type> &P, const Eigen::Vector<float_type, -1> &q, float_type c,
            const ADMM_settings<float_type> &settings, ADMM_solution<float_type>* solution) const
        {
            // check dimensions
            if (P.rows() != this->n || P.cols() != this->n || q.size() != this->n)
            {
                throw std::invalid_argument("Optimize over: inconsistent dimensions.");
            }

            // get cost matrices in factor space
            Eigen::SparseMatrix<float_type> P_fact = this->G.transpose()*P*this->G;
            Eigen::Vector<float_type, -1> q_fact = this->G.transpose()*(P*this->c + q);

            // bounds
            Eigen::Vector<float_type, -1> xi_lb, xi_ub;
            if (this->zero_one_form)
                xi_lb = Eigen::Vector<float_type, -1>::Zero(this->nG);
            else
                xi_lb = -1*Eigen::Vector<float_type, -1>::Ones(this->nG);
            xi_ub = Eigen::Vector<float_type, -1>::Ones(this->nG);

            // build ADMM object
            ADMM_solver<float_type> solver;
            solver.setup(P_fact, q_fact, this->A, this->b, xi_lb, xi_ub, settings, c);

            // solve
            ADMM_solution<float_type> sol = solver.solve();
            if (solution != nullptr)
                *solution = sol;

            // check feasibility and return solution
            if (sol.infeasible)
                return Eigen::Vector<float_type, -1>();
            else
                return this->G*sol.z + this->c;
        }

        Eigen::Vector<float_type, -1> project_point_admm(const Eigen::Vector<float_type, -1>& x,
            const ADMM_settings<float_type>& settings, ADMM_solution<float_type>* solution) const
        {
            // check dimensions
            if (this->n != x.size())
            {
                throw std::invalid_argument("Point projection: inconsistent dimensions.");
            }

            // build QP for ADMM
            Eigen::SparseMatrix<float_type> P = this->G.transpose()*this->G;
            Eigen::Vector<float_type, -1> q = this->G.transpose()*(this->c-x);
            
            Eigen::Vector<float_type, -1> xi_lb, xi_ub;
            if (this->zero_one_form)
                xi_lb = Eigen::Vector<float_type, -1>::Zero(this->nG);
            else
                xi_lb = -1*Eigen::Vector<float_type, -1>::Ones(this->nG);

            xi_ub = Eigen::Vector<float_type, -1>::Ones(this->nG);

            // build ADMM object
            ADMM_solver<float_type> solver;
            solver.setup(P, q, this->A, this->b, xi_lb, xi_ub, settings);

            // solve
            ADMM_solution<float_type> sol = solver.solve();
            if (solution != nullptr)
                *solution = sol;

            // check feasibility and return solution
            if (sol.infeasible)
                throw std::invalid_argument("Point projection: infeasible");
            else
                return this->G*sol.z + this->c;
        }

        bool is_empty_admm(const ADMM_settings<float_type>& settings, ADMM_solution<float_type>* solution) const
        {
            // trivial case
            if (this->n == 0)
                return true;

            // optimize over P=I, q=0
            Eigen::SparseMatrix<float_type> P (this->nG, this->nG);
            P.setIdentity();
            Eigen::Vector<float_type, -1> q = Eigen::Vector<float_type, -1>::Zero(this->nG);

            // bounds
            Eigen::Vector<float_type, -1> xi_lb, xi_ub;
            if (this->zero_one_form)
                xi_lb = Eigen::Vector<float_type, -1>::Zero(this->nG);
            else
                xi_lb = -1*Eigen::Vector<float_type, -1>::Ones(this->nG);

            xi_ub = Eigen::Vector<float_type, -1>::Ones(this->nG);

            // build ADMM object
            ADMM_solver<float_type> solver;
            solver.setup(P, q, this->A, this->b, xi_lb, xi_ub, settings);

            // solve
            ADMM_solution<float_type> sol = solver.solve();
            if (solution != nullptr)
                *solution = sol;

            // check infeasibility flag
            return sol.infeasible;
        }

        float_type support_admm(const Eigen::Vector<float_type, -1>& d,
            const ADMM_settings<float_type>& settings, ADMM_solution<float_type>* solution) const
        {
            // check dimensions
            if (this->n != d.size())
            {
                throw std::invalid_argument("Support: inconsistent dimensions.");
            }

            // build QP for ADMM
            Eigen::SparseMatrix<float_type> P (this->nG, this->nG);
            Eigen::Vector<float_type, -1> q = -this->G.transpose()*d;
            
            Eigen::Vector<float_type, -1> xi_lb, xi_ub;
            if (this->zero_one_form)
                xi_lb = Eigen::Vector<float_type, -1>::Zero(this->nG);
            else
                xi_lb = -1*Eigen::Vector<float_type, -1>::Ones(this->nG);

            xi_ub = Eigen::Vector<float_type, -1>::Ones(this->nG);

            // build ADMM object
            ADMM_solver<float_type> solver;
            solver.setup(P, q, this->A, this->b, xi_lb, xi_ub, settings);

            // solve
            ADMM_solution<float_type> sol = solver.solve();
            if (solution != nullptr)
                *solution = sol;

            // check feasibility and return solution
            if (sol.infeasible) // Z is empty
                throw std::invalid_argument("Support: infeasible");
            else
                return d.dot(this->G*sol.z + this->c);
        }

        bool contains_point_admm(const Eigen::Vector<float_type, -1>& x,
            const ADMM_settings<float_type>& settings, ADMM_solution<float_type>* solution) const
        {
            // check dimensions
            if (this->n != x.size())
            {
                throw std::invalid_argument("Contains point: inconsistent dimensions.");
            }

            // build QP for ADMM
            Eigen::SparseMatrix<float_type> P (this->nG, this->nG); // zeros
            Eigen::Vector<float_type, -1> q (this->nG);
            q.setZero(); // zeros
            Eigen::SparseMatrix<float_type> A = vcat(this->A, this->G);
            Eigen::Vector<float_type, -1> b (this->nC + this->n);
            b.segment(0, this->nC) = this->b;
            b.segment(this->nC, this->n) = x-this->c;

            Eigen::Vector<float_type, -1> xi_lb, xi_ub;
            if (this->zero_one_form)
                xi_lb = Eigen::Vector<float_type, -1>::Zero(this->nG);
            else
                xi_lb = -1*Eigen::Vector<float_type, -1>::Ones(this->nG);
            xi_ub = Eigen::Vector<float_type, -1>::Ones(this->nG);    
            
            // solve
            ADMM_solver<float_type> solver;
            solver.setup(P, q, A, b, xi_lb, xi_ub, settings);
            ADMM_solution<float_type> sol = solver.solve();
            if (solution != nullptr)
                *solution = sol;

            // check feasibility and return solution
            return !(sol.infeasible);
        }

        // bounding box
        std::unique_ptr<AbstractZono> bounding_box_admm(const ADMM_settings<float_type>& settings, ADMM_solution<float_type>* solution) const
        {
            // make sure dimension is at least 1
            if (this->n == 0)
            {
                throw std::invalid_argument("Bounding box: empty set");
            }

            // init search direction for bounding box
            Eigen::Vector<float_type, -1> d (this->n);
            d.setZero();

            // build QP for ADMM
            Eigen::SparseMatrix<float_type> P (this->nG, this->nG);
            Eigen::Vector<float_type, -1> q = -this->G.transpose()*d;
            
            Eigen::Vector<float_type, -1> xi_lb, xi_ub;
            if (this->zero_one_form)
                xi_lb = Eigen::Vector<float_type, -1>::Zero(this->nG);
            else
                xi_lb = -1*Eigen::Vector<float_type, -1>::Ones(this->nG);

            xi_ub = Eigen::Vector<float_type, -1>::Ones(this->nG);

            // build ADMM object
            ADMM_solver<float_type> solver;
            solver.setup(P, q, this->A, this->b, xi_lb, xi_ub, settings);

            // get support in all box directions
            std::vector<std::pair<float_type, float_type>> box_bounds; // declare
            ADMM_solution<float_type> sol; // declare
            float_type s_neg, s_pos;

            for (int i=0; i<this->n; i++)
            {
                // negative direction

                // update QP
                d.setZero();
                d(i) = -1;
                q = -this->G.transpose()*d;
                solver.update_q(q);

                // solve
                sol = solver.solve();
                if (sol.infeasible)
                    throw std::invalid_argument("Bounding box: Z is empty");
                else
                    s_neg = -d.dot(this->G*sol.z + this->c);

                // positive direction

                // update QP
                d.setZero();
                d(i) = 1;
                q = -this->G.transpose()*d;
                solver.update_q(q);

                // solve
                sol = solver.solve();
                if (sol.infeasible)
                    throw std::invalid_argument("Bounding box: Z is empty");
                else
                    s_pos = d.dot(this->G*sol.z + this->c);

                // store bounds
                box_bounds.push_back(std::make_pair(s_neg, s_pos));
            }

            // make generator matrix and center
            Eigen::SparseMatrix<float_type> G (this->n, this->n);
            Eigen::Vector<float_type, -1> c (this->n);
            std::vector<Eigen::Triplet<float_type>> triplets;

            int ind = 0;
            for (auto it=box_bounds.begin(); it!=box_bounds.end(); ++it)
            {
                triplets.push_back(Eigen::Triplet<float_type>(ind, ind, (it->second - it->first)/2));
                c(ind) = (it->second + it->first)/2;
                ind++;
            }
            G.setFromSortedTriplets(triplets.begin(), triplets.end());

            // return solution
            if (solution != nullptr)
                *solution = sol;

            // return zonotope
            return std::make_unique<Zono<float_type>>(G, c, false);  
        }




        // utilities

        // find unused generators
        std::vector<int> find_unused_generators(const Eigen::SparseMatrix<float_type>& G, const Eigen::SparseMatrix<float_type>& A)
        {
            std::vector<int> idx_no_cons;
            for (int k=0; k<A.outerSize(); k++)
            {
                bool is_unused = true;
                for (typename Eigen::SparseMatrix<float_type>::InnerIterator it(A, k); it; ++it)
                {
                    if (it.value() != 0)
                    {
                        is_unused = false;
                        break;
                    }
                }

                if (is_unused)
                {
                    idx_no_cons.push_back(k);
                }
            }

            // check if any of idx_no_cons multiply only zeros
            std::vector<int> idx_to_remove;
            for (auto it_idx=idx_no_cons.begin(); it_idx!=idx_no_cons.end(); ++it_idx)
            {
                bool is_zero = true;
                for (typename Eigen::SparseMatrix<float_type>::InnerIterator it(G, *it_idx); it; ++it)
                {
                    if (it.value() != 0)
                    {
                        is_zero = false;
                        break;
                    }
                }

                if (is_zero)
                {
                    idx_to_remove.push_back(*it_idx);
                }
            }

            return idx_to_remove;
        }

        // remove generators
        void remove_generators(Eigen::SparseMatrix<float_type>& G, Eigen::SparseMatrix<float_type>& A, const std::vector<int>& idx_to_remove)
        {
            // declare triplets
            std::vector<Eigen::Triplet<float_type>> triplets;

            // update G
            auto it_remove = idx_to_remove.begin();
            int delta_ind = 0;
            for (int k=0; k<G.outerSize(); k++)
            {
                if (k == *it_remove)
                {
                    delta_ind++;
                    it_remove++;
                    continue;
                }
                else
                {
                    for (typename Eigen::SparseMatrix<float_type>::InnerIterator it(G, k); it; ++it)
                    {
                        triplets.push_back(Eigen::Triplet<float_type>(it.row(), k-delta_ind, it.value()));
                    }
                }
            }
            G.resize(G.rows(), G.cols() - delta_ind);
            G.setFromTriplets(triplets.begin(), triplets.end());

            // update A
            triplets.clear();
            it_remove = idx_to_remove.begin();
            delta_ind = 0;
            for (int k=0; k<A.outerSize(); k++)
            {
                if (k == *it_remove)
                {
                    delta_ind++;
                    it_remove++;
                    continue;
                }
                else
                {
                    for (typename Eigen::SparseMatrix<float_type>::InnerIterator it(A, k); it; ++it)
                    {
                        triplets.push_back(Eigen::Triplet<float_type>(it.row(), k-delta_ind, it.value()));
                    }
                }
            }
            A.resize(A.rows(), A.cols() - delta_ind);
            A.setFromTriplets(triplets.begin(), triplets.end());
        }
};

} // namespace ZonoOpt

#endif