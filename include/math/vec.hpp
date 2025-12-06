#pragma once

#ifdef __CUDACC__
	#define HD __host__ __device__ //cpp files will see this as if cuda doesnt exist at all
#else
	#define HD
#endif

#include <cmath>
#include <iostream>

#include <concepts>

#include <bit>
#include <cstdint>


HD inline float fast_inverse(float p_magnitude) { //based on the quake algo
	float x2 = p_magnitude * 0.5f;
	float y = p_magnitude;

	std::uint32_t i;
	#ifdef __CUDA_ARCH__
		i = __float_as_uint(y); //gpu cast
	#else
		i = std::bit_cast<std::uint32_t>(y); //cpu cast
	#endif

	i = 0x5F3759DFu - (i >> 1);
	
	#ifdef __CUDA_ARCH__
		y = __uint_as_float(i); //gpu cast
	#else
		y = std::bit_cast<float>(i); //cpu cast
	#endif

	float threeHalfs = 1.5f; //adjust slightly for different speed testing
	y *= (threeHalfs - (x2 * y * y));
	y *= (threeHalfs - (x2 * y * y));
	y *= (threeHalfs - (x2 * y * y));

	return y;
}

template <typename t_type>
concept arithmetic = std::integral<t_type> || std::floating_point<t_type>; //constraint type must be a number basically

template <std::size_t t_dimension, arithmetic t_type>
struct vec;

template <std::size_t t_dimension, arithmetic t_type>
struct vec_data {
	t_type m_elem[t_dimension];
};

template<arithmetic t_type>
struct vec_data<1, t_type> {
	union { 
		struct { t_type x; }; 
		t_type m_elem[1];
	};
};

template<arithmetic t_type>
struct vec_data<2, t_type> {
	union { 
		struct { t_type x, y; }; 
		t_type m_elem[2];
	};
};

template<arithmetic t_type>
struct vec_data<3, t_type> {
	union { 
		struct { t_type x, y, z; }; 
		t_type m_elem[3];
	};
};

template<arithmetic t_type>
struct vec_data<4, t_type> {
	union { 
		struct { t_type x, y, z, w; }; 
		t_type m_elem[4];
	};
};


template <std::size_t t_dimension, arithmetic t_type>
struct vec_operation : public vec_data<t_dimension, t_type> {
	using data = vec_data<t_dimension, t_type>;
	using data::m_elem;

	HD vec_operation() {
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] = t_type{};
		}
	}

	template<typename t_other_type> //allows me to copy float into int (relying on static_cast handling conversions when needed)
	HD vec_operation(const vec<t_dimension, t_other_type> &p_other) { //enforces they are the same length at least
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] = static_cast<t_type>(p_other[dim]);
		}
	}

	HD vec_operation(t_type p_scale) {
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] = p_scale;
		}
	};

	template<typename... args>
	requires (sizeof...(args) == t_dimension && (std::convertible_to<args, t_type> && ...))
	HD vec_operation(args... p_args) {
		t_type values[] = { static_cast<t_type>(p_args)... };
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] = values[dim];
		}
	}

	HD constexpr double squared_magnitude() const {
		double len = 0;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			len += m_elem[dim] * m_elem[dim];
		}
		return len;
	};

	HD constexpr double magnitude() const {
		double length = squared_magnitude();
		return static_cast<double>(fast_inverse(static_cast<float>(length))) * length;
	}

	HD constexpr vec<t_dimension, t_type> unit() const {
		double length = magnitude();
		return static_cast<vec<t_dimension, t_type>>(*this) / (length != 0 ? length : 1.0); //avoid dividing by 0 (keep an eye out as this would cause a bug wont warn and instead catching a divide by 0)
	}

	HD constexpr auto dot(const vec<t_dimension, t_type> &p_other) const { //sums the pairwise product of all vector elements
		using promoted = std::conditional_t<std::is_integral_v<t_type>, long long, double>; //this will convert an int of t_type to long long to prevent overflow for large valued dots
		promoted result = promoted{};
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			result += static_cast<promoted>(m_elem[dim]) * static_cast<promoted>(p_other[dim]);
		}
		return result;
	}

	HD constexpr t_type min() const {
		t_type smallest = m_elem[0];
		for(std::size_t dim = 1; dim < t_dimension; dim++) {
			if(m_elem[dim] < smallest) {smallest = m_elem[dim];}
		}
		return smallest;
	}

	HD constexpr t_type max() const {
		t_type largest = m_elem[0];
		for(std::size_t dim = 1; dim < t_dimension; dim++) {
			if(m_elem[dim] > largest) {largest = m_elem[dim];}
		}
		return largest;
	}

	HD constexpr std::size_t minDimension() const {
		std::size_t smallest_dimension = 0;
		t_type smallest = m_elem[0];
		for(std::size_t dim = 1; dim < t_dimension; dim++) {
			if(m_elem[dim] > smallest) {
				smallest = m_elem[dim];
				smallest_dimension = dim;
			}
		}
		return smallest_dimension;
	}

	HD constexpr std::size_t maxDimension() const {
		std::size_t largest_dimension = 0;
		t_type largest = m_elem[0];
		for(std::size_t dim = 1; dim < t_dimension; dim++) {
			if(m_elem[dim] > largest) {
				largest = m_elem[dim];
				largest_dimension = dim;
			}
		}
		return largest_dimension;
	}

	HD constexpr vec<3, t_type> cross(const vec<3, t_type>& p_other) const requires (t_dimension == 3) { //neat little cpp20 that wont generate this function for non 3d vec
		return {
			m_elem[1] * p_other.m_elem[2] - m_elem[2] * p_other.m_elem[1],
			m_elem[2] * p_other.m_elem[0] - m_elem[0] * p_other.m_elem[2],
			m_elem[0] * p_other.m_elem[1] - m_elem[1] * p_other.m_elem[0]
		};
		// return {
		// 	y * p_other.z - z * p_other.y,
		// 	z * p_other.x - x * p_other.z,
		// 	x * p_other.y - y * p_other.x
		// };
	}

	template <typename t_cast>
	HD operator vec<t_dimension, t_cast>() const { //should enforce equal sizes
		vec<t_dimension, t_cast> result;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = static_cast<t_cast>(m_elem[dim]);}
		return result;
	}

	HD t_type operator[](int dim) const {return m_elem[dim];}
	HD t_type& operator[](int dim) {return m_elem[dim];}

	HD constexpr vec<t_dimension, t_type> operator-() const { //to negate the vec components
		vec<t_dimension, t_type> cpy;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			cpy[dim] = -m_elem[dim];
		}
		return cpy;
	}

	HD constexpr vec<t_dimension, t_type>& operator=(const vec<t_dimension, t_type> &p_other) {
		if(this == &p_other) {return *this;}
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] = p_other[dim];
		}
		return *this;
	}

	/* assignment operators */
	HD constexpr vec_operation& operator+=(const vec<t_dimension, t_type>& p_other) {
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] += p_other[dim];
		}
		return *this;
	};

	HD constexpr vec_operation& operator-=(const vec<t_dimension, t_type>& p_other) {
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] -= p_other[dim];
		}
		return *this;
	}

	HD constexpr vec_operation& operator*=(const vec<t_dimension, t_type>& p_other) {
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] *= p_other[dim];
		}
		return *this;
	}

	/*having internal discussions whether to route all compound assignment operators via the according binary arithmetic operator | for now ill leave it until im sure routing through another function call isnt going to slow me down */
	template<arithmetic t_arithmetic> //let me do a vec<3, float> * 0.5 (double literal) without making a big deal of it
	HD constexpr vec_operation& operator*=(const t_arithmetic p_scale) {
		const t_type scale = static_cast<t_type>(p_scale); //explicit cast to make code say exactly what im allowing for
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] *= scale;
		}
		return *this;
	}

	HD constexpr vec_operation& operator/=(const vec<t_dimension, t_type>& p_other) {
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] /= p_other[dim];
		}
		return *this;
	}

	template<arithmetic t_arithmetic>
	HD constexpr vec_operation& operator/=(const t_arithmetic p_scale) {
		t_type scale = static_cast<t_type>(p_scale);
		for(std::size_t dim = 0; dim < t_dimension; dim++) {
			m_elem[dim] /= scale;
		}
		return *this;
	}	


	HD friend constexpr vec<t_dimension, t_type> operator+(const vec<t_dimension, t_type>& _this, const vec<t_dimension, t_type>& p_other) {
		vec<t_dimension, t_type> result;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = _this[dim] + p_other[dim];}
		return result;
	};
	HD friend constexpr vec<t_dimension, t_type> operator-(const vec<t_dimension, t_type>& _this, const vec<t_dimension, t_type>& p_other) {
		vec<t_dimension, t_type> result;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = _this[dim] - p_other[dim];}
		return result;
	}
	HD friend constexpr vec<t_dimension, t_type> operator*(const vec<t_dimension, t_type>& _this, const vec<t_dimension, t_type>& p_other) {
		vec<t_dimension, t_type> result;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = _this[dim] * p_other[dim];}
		return result;
	}

	template<arithmetic t_arithmetic>
	HD friend constexpr vec<t_dimension, t_type> operator*(const vec<t_dimension, t_type>& _this, const t_arithmetic p_scale) {
		vec<t_dimension, t_type> result;
		t_type scale = static_cast<t_type>(p_scale);
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = _this[dim] * scale;}
		return result;
	}
	HD friend constexpr vec<t_dimension, t_type> operator*(const t_type p_scale, const vec<t_dimension, t_type>& _this) { //allow for 0.5 * vec not restricting to vec * 0.5
		return _this * p_scale;
	}
	HD friend constexpr vec<t_dimension, t_type> operator/(const vec<t_dimension, t_type>& _this, const vec<t_dimension, t_type>& p_other) {
		vec<t_dimension, t_type> result;
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = _this[dim] / p_other[dim];}
		return result;
	}

	template<arithmetic t_arithmetic>
	HD friend constexpr vec<t_dimension, t_type> operator/(const vec<t_dimension, t_type>& _this, const t_arithmetic p_scale) {
		vec<t_dimension, t_type> result;
		t_type scale = static_cast<t_type>(p_scale);
		for(std::size_t dim = 0; dim < t_dimension; dim++) {result[dim] = _this[dim] / scale;}
		return result;
	}

	#ifndef __CUDA_ARCH__ //disable << out as an operator as cuda doesnt support any host terminal ostream stuff from cpp
	friend inline std::ostream& operator<<(std::ostream& out, const vec<t_dimension, t_type>& v) {
		out << '(';
		for(std::size_t dim = 0; dim < t_dimension-1; dim++) {
			out << v[dim] << ", ";
		}
		out << v[t_dimension-1] << ')';
		return out;
	}
	#endif
};

template <std::size_t t_dimension, arithmetic t_type>
struct vec : public vec_operation<t_dimension, t_type> {
	using value_type = t_type; //expose the template values if users ever want them
	static constexpr std::size_t dimension_count = t_dimension;
	
	using operators = vec_operation<t_dimension, t_type>;
	using operators::m_elem; //allow visibility to the data
	using operators::vec_operation; //allowing visibility to the constructors
};



//native supported vectors as "regular types". dim can use these similar to int and double but allowing for N dimensioned vectors

enum class dimension : int {
	x = 0,
	y = 1,
	z = 2,
	w = 4
};

using vec1f = vec<1, float>;
using vec1i = vec<1, int>;

using vec2f = vec<2, float>;
using vec2i = vec<2, int>;

using vec3f = vec<3, float>;
using vec3d = vec<3, double>;
using vec3i = vec<3, int>;

using vec4f= vec<4, float>;
using vec4i = vec<4, int>;