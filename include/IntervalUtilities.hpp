#ifndef __ZONOOPT_INTERVAL_UTILITIES_HPP__
#define __ZONOOPT_INTERVAL_UTILITIES_HPP__

#include <limits>
#include <string>
#include <ostream>
#include <vector>
#include <exception>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/*
Reference: 
"Applied Interval Analysis"
Luc Jaulin, Michel Kieffer, Olivier Didrit, Eric Walter
*/

namespace ZonoOpt
{

template <typename float_type>
class Interval
{
    public:

        // constructor
        Interval(): y_min(0), y_max(0) {}
        Interval(float_type y_min, float_type y_max): y_min(y_min), y_max(y_max) {}

        // methods
        Interval<float_type> operator+(const Interval<float_type>& other) const
        {
            return Interval<float_type>(y_min + other.y_min, y_max + other.y_max);
        }

        Interval<float_type> operator-(const Interval<float_type>& other) const
        {
            return Interval<float_type>(y_min - other.y_max, y_max - other.y_min);
        }

        Interval<float_type> operator*(const Interval<float_type>& other) const
        {
            float_type a = y_min * other.y_min;
            float_type b = y_min * other.y_max;
            float_type c = y_max * other.y_min;
            float_type d = y_max * other.y_max;

            return Interval<float_type>(std::min(std::min(a, b), std::min(c, d)), std::max(std::max(a, b), std::max(c, d)));
        }

        Interval<float_type> operator*(float_type alpha) const
        {
            if (alpha >= 0)
                return Interval<float_type>(y_min * alpha, y_max * alpha);
            else
                return Interval<float_type>(y_max * alpha, y_min * alpha);
        }

        Interval<float_type> inv() const
        {
            if (y_min == 0 && y_max == 0)
                return Interval<float_type>(std::numeric_limits<float_type>::infinity(), -std::numeric_limits<float_type>::infinity()); // empty set
            else if (y_min > 0 || y_max < 0)
                return Interval<float_type>(1/y_max, 1/y_min);
            else if (y_min == 0 && y_max > 0)
                return Interval<float_type>(1/y_max, std::numeric_limits<float_type>::infinity());
            else if (y_min < 0 && y_max == 0)
                return Interval<float_type>(-std::numeric_limits<float_type>::infinity(), 1/y_min);
            else
                return Interval<float_type>(-std::numeric_limits<float_type>::infinity(), std::numeric_limits<float_type>::infinity());
        }

        Interval<float_type> operator/(const Interval<float_type>& other) const
        {
            return *this * other.inv();
        }

        bool is_empty() const
        {
            return y_min > y_max;
        }

        bool contains(float_type y) const
        {
            return (y >= y_min) && (y <= y_max);
        }

        std::string print() const
        {
            return "[" + std::to_string(y_min) + ", " + std::to_string(y_max) + "]";
        }

        friend std::ostream& operator<<(std::ostream& os, const Interval<float_type>& interval)
        {
            os << interval.print();
            return os;
        }

        // members
        float_type y_min, y_max;

}; // end class Interval


template <typename float_type>
class Box
{
    public:  
        Box() = default;
        Box(size_t size): vals(size) {}
        Box(const std::vector<Interval<float_type>>& vals): vals(vals) {}
        
        void push_back(const Interval<float_type>& val)
        {
            vals.push_back(val);
        }

        void pop_back()
        {
            vals.pop_back();
        }

        void clear()
        {
            vals.clear();
        }

        Interval<float_type> operator[](int i) const
        {
            return vals[i];
        }

        size_t size() const
        {
            return vals.size();
        }

        // TO DO: implement using Eigen::Matrix<Interval<float_type>, -1, -1>
        Box<float_type> linear_map(const Eigen::Matrix<float_type, -1, -1>& A) const
        {
            // input handling
            if (A.cols() != vals.size())
                throw std::invalid_argument("Matrix A must have the same number of columns as the size of the Box");

            // declare
            Box<float_type> y(A.rows());
  
            // linear map
            for (int i=0; i<A.rows(); i++)
            {
                y.vals[i] = Interval<float_type>(0, 0);
                for (int j=0; j<A.cols(); j++)
                {
                    y.vals[i] = y.vals[i] + (this->vals[j]*A(i, j));
                }
            }
            return y;
        }

        // TO DO: implement using Eigen::SparseMatrix<Interval<float_type>>
        Box<float_type> linear_map(const Eigen::SparseMatrix<float_type, Eigen::RowMajor>& A) const
        {
            // input handling
            if (A.cols() != vals.size())
                throw std::invalid_argument("Matrix A must have the same number of columns as the size of the Box");

            // declare
            Box<float_type> y (A.rows());
            
            // linear map
            for (int i=0; i<A.rows(); i++)
            {
                y.vals[i] = Interval<float_type>(0, 0);
                for (typename Eigen::SparseMatrix<float_type, Eigen::RowMajor>::InnerIterator it(*A, i); it; ++it)
                {
                    y.vals[i] = y.vals[i] + (this->vals[it.col()]*it.value());
                }
            }
            return y;
        }

        Interval<float_type> dot(const Eigen::Vector<float_type, -1>& x) const
        {
            // input handling
            if (x.size() != vals.size())
                throw std::invalid_argument("Vector x must have the same size as the Box");

            // declare
            Interval<float_type> y(0, 0);
            
            // linear map
            for (int i=0; i<this->vals.size(); i++)
                y = y + (this->vals[i]*x(i));
            return y;
        }

    private:

        // members
        std::vector<Interval<float_type>> vals;

};

} // namespace ZonoOpt

#endif