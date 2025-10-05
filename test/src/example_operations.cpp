#include <iostream>
#include <exception>

#define zono_float float
#include "ZonoOpt.hpp"
using namespace ZonoOpt;

int main()
{
    // settings struct
    OptSettings settings;
    settings.verbose = true;
    settings.n_threads_bnb = 1;

    // create conzono
    Eigen::SparseMatrix<float> G(2, 2);
    G.insert(0, 0) = 1;
    G.insert(1, 1) = 1;

    Eigen::Vector<float, -1> c(2);
    c.setZero();

    Eigen::SparseMatrix<float> A(1, 2); 
    A.insert(0, 0) = 1;

    Eigen::Vector<float, -1> b(1);
    b(0) = 1;

    ConZono Z1 (G, c, A, b, true);
    Zono Z2 (G, c, false);

    // try minkowski sum
    ZonoPtr Z_sum = minkowski_sum(Z1, Z2);
    std::cout << "Z_sum is conzono?: " << Z_sum->is_conzono() << std::endl;
    std::cout << "Z_sum: " << *Z_sum << std::endl;

    // union
    std::vector<HybZono*> Zs;
    Zs.push_back(&Z1);
    Zs.push_back(&Z2);

    ZonoPtr U = union_of_many(Zs);
    std::cout << "U: " << *U << std::endl;
    std::cout << "U is empty? " << U->is_empty(settings) << std::endl;
    
    // convex relaxation
    auto C = U->convex_relaxation();
    std::cout << "C: " << *C << std::endl;

    // project point
    Eigen::VectorXf x (C->get_n());
    x.setOnes();
    
    std::cout << "Point projection onto C: " << std::endl;
    std::cout << C->project_point(x, settings) << std::endl;

    std::cout << "Point projection onto U: " << std::endl;
    std::cout << U->project_point(x, settings) << std::endl;


    // build hybzono with redundant constraints
    Eigen::SparseMatrix<float> Gc_h(2, 2);
    Gc_h.setIdentity();
    Eigen::SparseMatrix<float> Gb_h(2, 2);
    Gb_h.setIdentity();
    Eigen::Vector<float, 2> c_h;
    c_h.setZero();

    Eigen::MatrixXf Acd_h = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf Abd_h = Eigen::MatrixXf::Ones(2, 2);
    Eigen::SparseMatrix<float> Ac_h = Acd_h.sparseView();
    Eigen::SparseMatrix<float> Ab_h = Abd_h.sparseView();
    Eigen::Vector<float, 2> b_h;
    b_h << 1, 1;

    HybZono Zh (Gc_h, Gb_h, c_h, Ac_h, Ab_h, b_h, true);
    Zh.remove_redundancy();
    
    std::cout << "Zh: " << Zh << std::endl;

    return 0;
}