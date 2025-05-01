#ifndef __ZONOOPT_HYBZONO_HPP__
#define __ZONOOPT_HYBZONO_HPP__

#include "SparseMatrixUtilities.hpp"
#include "AbstractZono.hpp"
#include <exception>

namespace ZonoOpt
{

template <typename float_type>
class HybZono : public AbstractZono<float_type>
{
    public:
        
        // constructors
        HybZono() = default;
        HybZono(const Eigen::SparseMatrix<float_type>& Gc, const Eigen::SparseMatrix<float_type>& Gb, const Eigen::Vector<float_type, -1>& c,
            const Eigen::SparseMatrix<float_type>& Ac, const Eigen::SparseMatrix<float_type>& Ab,  const Eigen::Vector<float_type, -1>& b,
            bool zero_one_form=false)
        {
            set(Gc, Gb, c, Ac, Ab, b, zero_one_form);
        } 

        // set methods
        void set(const Eigen::SparseMatrix<float_type>& Gc, const Eigen::SparseMatrix<float_type>& Gb, const Eigen::Vector<float_type, -1>& c,
            const Eigen::SparseMatrix<float_type>& Ac, const Eigen::SparseMatrix<float_type>& Ab, const Eigen::Vector<float_type, -1>& b,
            bool zero_one_form=false)
        {
            // check dimensions
            if (Gc.rows() != c.size() || Gb.rows() != c.size() || Gc.cols() != Ac.cols() 
            || Gb.cols() != Ab.cols() || Ac.rows() != b.size() || Ab.rows() != b.size())
            {
            throw std::invalid_argument("HybZono: inconsistent dimensions.");
            }

            this->Gc = Gc;
            this->Gb = Gb;
            this->Ac = Ac;
            this->Ab = Ab;
            this->c = c;
            this->b = b;
            this->nGc = Gc.cols();
            this->nGb = Gb.cols();
            this->nC = Ac.rows();
            this->n = Gc.rows();
            this->zero_one_form = zero_one_form;

            make_G_A();
        }

        // get methods
        int get_n() const { return this->n; }
        int get_nC() const { return this->nC; }
        int get_nG() const { return this->nG; }
        int get_nGc() const { return this->nGc; }
        int get_nGb() const { return this->nGb; }
        Eigen::SparseMatrix<float_type> get_Gc() const { return this->Gc; }
        Eigen::SparseMatrix<float_type> get_Gb() const { return this->Gb; }
        Eigen::SparseMatrix<float_type> get_G() const { return this->G; }
        Eigen::SparseMatrix<float_type> get_Ac() const { return this->Ac; }
        Eigen::SparseMatrix<float_type> get_Ab() const { return this->Ab; }
        Eigen::SparseMatrix<float_type> get_A() const { return this->A; }
        Eigen::Vector<float_type, -1> get_c() const { return this->c; }
        Eigen::Vector<float_type, -1> get_b() const { return this->b; }
        bool is_0_1_form() const { return this->zero_one_form; }


        // generator conversion between [-1,1] and [0,1]
        void convert_form()
        {
            Eigen::Vector<float_type, -1> c, b;
            Eigen::SparseMatrix<float_type> Gb, Ab, Ac, Gc;

            if (!this->zero_one_form) // convert to [0,1] generators
            {
                c = this->c - this->G*Eigen::Vector<float_type, -1>::Ones(this->nG);
                b = this->b + this->A*Eigen::Vector<float_type, -1>::Ones(this->nG);
                Gb = 2*this->Gb;
                Ab = 2*this->Ab;
                Gc = 2*this->Gc;
                Ac = 2*this->Ac;

                set(Gc, Gb, c, Ac, Ab, b, true);
            }
            else // convert to [-1,1] generators
            {
                c = this->c + 0.5*this->G*Eigen::Vector<float_type, -1>::Ones(this->nG);
                b = this->b - 0.5*this->A*Eigen::Vector<float_type, -1>::Ones(this->nG);
                Gb = 0.5*this->Gb;
                Ab = 0.5*this->Ab;
                Gc = 0.5*this->Gc;
                Ac = 0.5*this->Ac;

                set(Gc, Gb, c, Ac, Ab, b, false);
            }
        }


        // remove redundancy
        void remove_redundancy() override
        {
            // remove redundant constraints
            remove_redundant<float_type>(this->A, this->b);
            this->nC = this->A.rows();

            // update Ac, Ab
            set_Ac_Ab_from_A();
            
            // identify unused generators
            std::vector<int> idx_c_to_remove = this->find_unused_generators(this->Gc, this->Ac);
            std::vector<int> idx_b_to_remove = this->find_unused_generators(this->Gb, this->Ab);

            // remove generators
            if (idx_c_to_remove.size() != 0)
            {
                this->remove_generators(this->Gc, this->Ac, idx_c_to_remove);
            }
            if (idx_b_to_remove.size() != 0)
            {
                this->remove_generators(this->Gb, this->Ab, idx_b_to_remove);
            }

            // update equivalent matrices
            make_G_A();

            // update number of generators
            this->nG = this->G.cols();
            this->nGc = this->Gc.cols();
            this->nGb = this->Gb.cols();
        }

        // convex relaxation
        ConZono<float_type> convex_relaxation() const
        {
            return ConZono<float_type>(this->G, this->c, this->A, this->b, this->zero_one_form);
        }

        
        // display methods
        std::string print() const
        {
            std::stringstream ss;
            ss << "HybZono: " << std::endl;
            ss << "n: " << this->n << std::endl;
            ss << "nGc: " << this->nGc << std::endl;
            ss << "nGb: " << this->nGb << std::endl;
            ss << "nC: " << this->nC << std::endl;
            ss << "Gc: " << Eigen::Matrix<float_type, -1, -1>(this->Gc) << std::endl;
            ss << "Gb: " << Eigen::Matrix<float_type, -1, -1>(this->Gb) << std::endl;
            ss << "c: " << this->c << std::endl;
            ss << "Ac: " << Eigen::Matrix<float_type, -1, -1>(this->Ac) << std::endl;
            ss << "Ab: " << Eigen::Matrix<float_type, -1, -1>(this->Ab) << std::endl;
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
            throw std::invalid_argument("Optimize over: not implemented for hybrid zonotope.");
        }

        // project point
        Eigen::Vector<float_type, -1> project_point(const Eigen::Vector<float_type, -1>& x, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(), 
            ADMM_solution<float_type>* solution=nullptr) const
        {
            throw std::invalid_argument("Point projection not implemented for HybZono.");
        }

        // is empty
        bool is_empty(const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            throw std::invalid_argument("Is empty not implemented for HybZono.");
        }

        // support
        float_type support(const Eigen::Vector<float_type, -1>& d, 
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            throw std::invalid_argument("Support not implemented for HybZono.");
        }

        // contains point
        bool contains_point(const Eigen::Vector<float_type, -1>& x,
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            throw std::invalid_argument("Contains point not implemented for HybZono.");
        }

        // bounding box - NOT NECESSARILY TIGHT
        std::unique_ptr<AbstractZono<float_type>> bounding_box(
            const ADMM_settings<float_type> &settings=ADMM_settings<float_type>(),
            ADMM_solution<float_type>* solution=nullptr) const
        {
            return AbstractZono<float_type>::bounding_box_admm(settings, solution);
        }

    protected:

        // make G, A
        void make_G_A()
        {
            std::vector<Eigen::Triplet<float_type>> tripvec;
            get_triplets_offset(this->Gc, tripvec, 0, 0);
            get_triplets_offset(this->Gb, tripvec, 0, this->nGc);
            this->G.resize(this->n, this->nGc + this->nGb);
            this->G.setFromTriplets(tripvec.begin(), tripvec.end());

            tripvec.clear();
            get_triplets_offset(this->Ac, tripvec, 0, 0);
            get_triplets_offset(this->Ab, tripvec, 0, this->nGc);
            this->A.resize(this->nC, this->nGc + this->nGb);
            this->A.setFromTriplets(tripvec.begin(), tripvec.end());

            this->nG = this->nGc + this->nGb;
        }
            
        // methods
        void set_Ac_Ab_from_A()
        {
            std::vector<Eigen::Triplet<float_type>> triplets_Ac, triplets_Ab;

            // iterate over A
            for (int k=0; k<this->A.outerSize(); ++k)
            {
                for (typename Eigen::SparseMatrix<float_type>::InnerIterator it(this->A, k); it; ++it)
                {
                    if (it.col() < this->nGc)
                    {
                        triplets_Ac.push_back(Eigen::Triplet<float_type>(it.row(), it.col(), it.value()));
                    }
                    else
                    {
                        triplets_Ab.push_back(Eigen::Triplet<float_type>(it.row(), it.col()-this->nGc, it.value()));
                    }
                }
            }

            // set Ac, Ab
            this->Ac.resize(this->nC, this->nGc);
            this->Ac.setFromTriplets(triplets_Ac.begin(), triplets_Ac.end());
            this->Ab.resize(this->nC, this->nGb);
            this->Ab.setFromTriplets(triplets_Ab.begin(), triplets_Ab.end());
        }
};


// setup functions
template <typename float_type>
HybZono<float_type> vrep_2_hybzono(const std::vector<Eigen::Matrix<float_type, -1, -1>> &Vpolys)
{
    // error handling
    if (Vpolys.size() == 0)
    {
        throw std::invalid_argument("set_from_vrep: Vpolys must have at least one polytope.");
    }

    // dimensions
    int n_polys = Vpolys.size();
    int n_dims = Vpolys[0].cols();
    int n_verts; // declare

    // check if all polytopes have the same number of dimensions
    for (auto it=Vpolys.begin(); it!=Vpolys.end(); ++it)
    {
        if (it->cols() != n_dims)
        {
            throw std::invalid_argument("set_from_vrep: all polytopes must have the same number of dimensions.");
        }
    }

    // initialize V and M matrices as std::vectors
    // each entry is a row
    std::vector<Eigen::Matrix<float_type, 1, -1>> V_vec, M_vec;
    Eigen::Matrix<float_type, 1, -1> M_row (n_polys);

    // loop through each polytope
    for (int i=0; i<n_polys; i++)
    {
        n_verts = Vpolys[i].rows();
        for (int j=0; j<n_verts; j++)
        {
            // check if the vertex is already in V_vec
            auto it_V = std::find(V_vec.begin(), V_vec.end(), Vpolys[i].row(j));
            if (it_V == V_vec.end())
            {
                V_vec.push_back(Vpolys[i].row(j));
                M_row.setZero();
                M_row(i) = 1;
                M_vec.push_back(M_row);
            }
            else
            {
                int idx = std::distance(V_vec.begin(), it_V);
                M_vec[idx](i) = 1;
            }
        }
    }

    int nV = V_vec.size(); // number of unique vertices

    // convert to Eigen matrices
    Eigen::Matrix<float_type, -1, -1> V (n_dims, nV);
    Eigen::Matrix<float_type, -1, -1> M (nV, n_polys);
    for (int i=0; i<V_vec.size(); i++)
    {
        V.col(i) = V_vec[i];
        M.row(i) = M_vec[i];
    }

    // directly build hybzono in [0,1] form

    // declare
    std::vector<Eigen::Triplet<float_type>> tripvec;

    // Gc = [V, 0]
    Eigen::SparseMatrix<float_type> Gc = V.sparseView();
    Gc.conservativeResize(n_dims, 2*nV);

    // Gb = [0]
    Eigen::SparseMatrix<float_type> Gb (n_dims, n_polys);

    // c = 0
    Eigen::Vector<float_type, -1> c (n_dims);
    c.setZero();

    // Ac = [1^T, 0^T;
    //       0^T, 0^T;
    //       I, diag[sum(M, 2)]]
    Eigen::SparseMatrix<float_type> Ac (2+nV, 2*nV);
    Eigen::SparseMatrix<float_type> I_nv (nV, nV);
    I_nv.setIdentity();
    for (int i=0; i<nV; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(0, i, 1));
    }
    get_triplets_offset(I_nv, tripvec, 2, 0);
    Eigen::Vector<float_type, -1> sum_M = M.rowwise().sum();
    for (int i=0; i<nV; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(2+i, nV+i, sum_M(i)));
    }
    Ac.setFromTriplets(tripvec.begin(), tripvec.end()); 

    // Ab = [0^T;
    //       1^T;
    //       -M]
    Eigen::SparseMatrix<float_type> Ab (2+nV, n_polys);
    tripvec.clear();
    for (int i=0; i<n_polys; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(1, i, 1));
    }
    Eigen::SparseMatrix<float_type> mM_sp = -M.sparseView();
    get_triplets_offset(mM_sp, tripvec, 2, 0);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());

    // b = [1;
    //      1;
    //      0]
    Eigen::Vector<float_type, -1> b (2+nV);
    b.setZero();
    b(0) = 1;
    b(1) = 1;

    // return hybzono
    return HybZono<float_type>(Gc, Gb, c, Ac, Ab, b, true);
}

template <typename float_type>
HybZono<float_type> zono_union_2_hybzono(std::vector<AbstractZono<float_type>*> &Zs)
{
    // can't be empty
    if (Zs.size() == 0)
    {
        throw std::invalid_argument("Zono union: empty input vector.");
    }

    // zonotope dimension
    int n_dims = Zs[0]->n;
    int n_zonos = Zs.size();

    // loop through Zs
    for (auto it=Zs.begin(); it!=Zs.end(); ++it)
    {
        // make sure all Zs are zonotopes or points
        if (!((*it)->is_zono() || (*it)->is_point()))
        {
            throw std::invalid_argument("Zono union: all inputs must be zonotopes.");
        }

        // make sure dimensions are consistent
        if ((*it)->n != n_dims)
        {
            throw std::invalid_argument("Zono union: inconsistent dimensions.");
        }

        // convert to [0,1] form
        if (!(*it)->zero_one_form)
        {
            (*it)->convert_form();
        }
    }

    // get unique generators and incidence matrix

    // initialize S and M matrices as std::vectors
    // each entry is a row
    int n_gens;
    std::vector<Eigen::Matrix<float_type, 1, -1>> S_vec, M_vec;
    Eigen::Matrix<float_type, 1, -1> M_row (n_zonos);
    Eigen::Matrix<float_type, -1, -1> Gd;

    // loop through each polytope
    for (int i=0; i<n_zonos; i++)
    {
        n_gens = Zs[i]->nG;
        Gd = Zs[i]->G.toDense();
        for (int j=0; j<n_gens; j++)
        {
            // check if the generator is already in V_vec
            auto it_S = std::find(S_vec.begin(), S_vec.end(), Gd.col(j));
            if (it_S == S_vec.end())
            {
                S_vec.push_back(Gd.col(j));
                M_row.setZero();
                M_row(i) = 1;
                M_vec.push_back(M_row);
            }
            else
            {
                int idx = std::distance(S_vec.begin(), it_S);
                M_vec[idx](i) = 1;
            }
        }
    }

    int nG = S_vec.size(); // number of unique generators

    // convert to Eigen matrices
    Eigen::Matrix<float_type, -1, -1> S (n_dims, nG);
    Eigen::Matrix<float_type, -1, -1> M (nG, n_zonos);
    for (int i=0; i<nG; i++)
    {
        S.col(i) = S_vec[i];
        M.row(i) = M_vec[i];
    }

    // directly build hybzono in [0,1] form

    // declare
    std::vector<Eigen::Triplet<float_type>> tripvec;

    // Gc = [S, 0]
    Eigen::SparseMatrix<float_type> Gc = S.sparseView();
    Gc.conservativeResize(n_dims, 2*nG);

    // Gb = [c]
    Eigen::Matrix<float_type, -1, -1> Gb_d (n_dims, n_zonos);
    for (int i=0; i<n_zonos; i++)
    {
        Gb_d.col(i) = Zs[i]->c;
    }
    Eigen::SparseMatrix<float_type> Gb = Gb_d.sparseView();

    // c = 0
    Eigen::Vector<float_type, -1> c (n_dims);
    c.setZero();

    // Ac = [0^T, 0^T;
    //       I, diag[sum(M, 2)]]
    Eigen::SparseMatrix<float_type> Ac (1+nG, 2*nG);
    Eigen::SparseMatrix<float_type> I_ng (nG, nG);
    I_ng.setIdentity();
    get_triplets_offset(I_ng, tripvec, 1, 0);
    Eigen::Vector<float_type, -1> sum_M = M.rowwise().sum();
    for (int i=0; i<nG; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(1+i, nG+i, sum_M(i)));
    }
    Ac.setFromTriplets(tripvec.begin(), tripvec.end()); 

    // Ab = [1^T;
    //       -M]
    Eigen::SparseMatrix<float_type> Ab (1+nG, n_zonos);
    tripvec.clear();
    for (int i=0; i<n_zonos; i++)
    {
        tripvec.push_back(Eigen::Triplet<float_type>(0, i, 1));
    }
    Eigen::SparseMatrix<float_type> mM_sp = -M.sparseView();
    get_triplets_offset(mM_sp, tripvec, 1, 0);
    Ab.setFromTriplets(tripvec.begin(), tripvec.end());

    // b = [1;
    //      0]
    Eigen::Vector<float_type, -1> b (1+nG);
    b.setZero();
    b(0) = 1;

    // return hybzono
    return HybZono<float_type>(Gc, Gb, c, Ac, Ab, b, true);
}


} // end namespace ZonoOpt

#endif