/* Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "impl.hpp"

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/hana/at.hpp>
#include <boost/hana/drop_back.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/throw_exception.hpp>
#include <edamer/detail/scalar.hpp>
#include <edamer/dt/el_complex.hpp>
#include <edamer/dt/exception.hpp>
#include <hbrs/mpl/dt/el_vector/impl.hpp>
#include <hbrs/mpl/dt/exception.hpp>
#include <pybind11/numpy.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

EDAMER_NAMESPACE_BEGIN(/* unnamed */)

EDAMER_NAMESPACE_BEGIN(detail)

template<typename Ring, typename Orientation>
auto
view_from_numpy_1d(
	py::array_t<Ring, py::array::f_style> & array,
	hana::basic_type<Orientation> = hana::type_c<Orientation>
) {
	static_assert(
		std::is_same_v<Orientation, mpl::el_column_vector_tag> ||
		std::is_same_v<Orientation, mpl::el_row_vector_tag>,
		"Only hbrs::mpl::el_column_vector_tag and hbrs::mpl::el_row_vector_tag are supported");
	
	if (array.ndim() != 1) {
		BOOST_THROW_EXCEPTION((incompatible_ndarray_exception{} << errinfo_ndarray_ndim{array.ndim()}));
	}
	
	py::buffer_info buf = array.request(array.writeable());
	El::Int m;
	El::Int n;
	
	if constexpr (std::is_same_v<Orientation, mpl::el_column_vector_tag>) {
		m = boost::numeric_cast<El::Int>( array.shape(0) );
		n = 1;
	} else {
		m = 1;
		n = boost::numeric_cast<El::Int>( array.shape(0) );
	}
	El::Int ldim = m;
	
	// Wrap matrix types in py::object instances because it is not possible in C++ to return different types.
	if constexpr (!std::is_const_v<Ring>) {
		if (array.writeable()) {
			return El::Matrix<Ring>{m, n, static_cast<Ring*>(buf.ptr), ldim};
		}
	}
	
	auto && matrix = El::Matrix<std::remove_const_t<Ring>>{m, n, static_cast<Ring const*>(buf.ptr), ldim};
	BOOST_ASSERT(matrix.Locked());
	return HBRS_MPL_FWD(matrix);
}

EDAMER_NAMESPACE_END(detail)

template<typename Ring, typename Orientation>
struct view_from_numpy_1d_t {
	static auto
	apply(py::array_t<Ring, py::array::f_style> & array) {
		static_assert(
			std::is_same_v<Orientation, mpl::el_column_vector_tag> ||
			std::is_same_v<Orientation, mpl::el_row_vector_tag>,
			"Only hbrs::mpl::el_column_vector_tag and hbrs::mpl::el_row_vector_tag are supported");
		
		auto matrix = detail::view_from_numpy_1d<Ring, Orientation>(array);
		
		if constexpr (std::is_same_v<Orientation, mpl::el_column_vector_tag>) {
			if (matrix.Width() == 1) {
				// Wrap matrix types in py::object instances because it is not possible in C++ to return different types.
				if (array.writeable()) {
					return py::cast(mpl::el_column_vector<Ring>{std::move(matrix)});
				} else {
					return py::cast(mpl::el_column_vector<Ring const>{std::move(matrix)});
				}
			}
		} else {
			if (matrix.Height() == 1) {
				// Wrap matrix types in py::object instances because it is not possible in C++ to return different types.
				if (array.writeable()) {
					return py::cast(mpl::el_row_vector<Ring>{std::move(matrix)});
				} else {
					return py::cast(mpl::el_row_vector<Ring const>{std::move(matrix)});
				}
			}
		}
		
		BOOST_THROW_EXCEPTION((
			mpl::incompatible_matrix_exception{} << mpl::errinfo_el_matrix_size{{matrix.Height(), matrix.Width()}}
		));
	}
};

EDAMER_NAMESPACE_BEGIN(detail)

template<typename Ring, typename Orientation>
auto
view_to_numpy_1d(
	El::Matrix<Ring> & matrix,
	py::handle base,
	hana::basic_type<Orientation> = hana::type_c<Orientation>
) {
	static_assert(
		std::is_same_v<Orientation, mpl::el_column_vector_tag> ||
		std::is_same_v<Orientation, mpl::el_row_vector_tag>,
		"Only hbrs::mpl::el_column_vector_tag and hbrs::mpl::el_row_vector_tag are supported");
	
	py::array::ShapeContainer shape;
	py::array::StridesContainer strides;
	
	if constexpr (std::is_same_v<Orientation, mpl::el_column_vector_tag>) {
		shape = py::array::ShapeContainer{ matrix.Height(), };
		strides = py::array::StridesContainer{ sizeof(Ring) };
	} else {
		shape = py::array::ShapeContainer{ matrix.Width(), };
		strides = py::array::StridesContainer{ sizeof(Ring) };
	}
	
	void * buf_ptr = matrix.Locked()
		? const_cast<void*>(static_cast<void const*>(matrix.LockedBuffer()))
		  /* is safe because it will be casted to const void* in py::array_t */
		: static_cast<void*>(matrix.Buffer());
	
	auto buf = py::buffer_info(
		buf_ptr,                               /* Pointer to buffer */
		sizeof(Ring),                          /* Size of one scalar */
		py::format_descriptor<Ring>::format(), /* Python struct-style format descriptor */
		1,                                     /* Number of dimensions */
		shape,                                 /* Buffer dimensions */
		strides,                               /* Strides (in bytes) for each index */
		matrix.Locked()                        /* Buffer is readonly */
	);
	
	auto && array = py::array_t<Ring, py::array::f_style>{buf, base};
	
	if (matrix.Locked()) {
		// Uses pybind11's private API to remove writeable flag because there is currently no official API for that
		// Ref.: https://github.com/pybind/pybind11/issues/481
		//TODO: Replace private API usage once it's supported!
		py::detail::array_proxy(array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
	}
	
	return HBRS_MPL_FWD(array);
}
EDAMER_NAMESPACE_END(detail)

template<typename Ring, typename Orientation>
struct view_to_numpy_1d_t {
	static auto
	apply(py::object &obj) {
		static_assert(
			std::is_same_v<Orientation, mpl::el_column_vector_tag> ||
			std::is_same_v<Orientation, mpl::el_row_vector_tag>,
			"Only hbrs::mpl::el_column_vector_tag and hbrs::mpl::el_row_vector_tag are supported");
		
		if constexpr (std::is_same_v<Orientation, mpl::el_column_vector_tag>) {
			mpl::el_column_vector<Ring> & vector = obj.cast<mpl::el_column_vector<Ring>&>();
			return detail::view_to_numpy_1d<std::remove_const_t<Ring>, Orientation>(vector.data(), obj);
		} else {
			mpl::el_row_vector<Ring> & vector = obj.cast<mpl::el_row_vector<Ring>&>();
			return detail::view_to_numpy_1d<std::remove_const_t<Ring>, Orientation>(vector.data(), obj);
		}
	}
};

auto scalars = hana::drop_back(hana::make_tuple(
	#ifdef EDAMER_ENABLE_SCALAR_INT
		EDAMER_TYPE_NAME_PAIRS(std::int32_t),
		EDAMER_TYPE_NAME_PAIRS(std::int32_t const),
		EDAMER_TYPE_NAME_PAIRS(std::uint32_t),
		EDAMER_TYPE_NAME_PAIRS(std::uint32_t const),
		EDAMER_TYPE_NAME_PAIRS(std::int64_t),
		EDAMER_TYPE_NAME_PAIRS(std::int64_t const),
		EDAMER_TYPE_NAME_PAIRS(std::uint64_t),
		EDAMER_TYPE_NAME_PAIRS(std::uint64_t const),
	#endif /* EDAMER_ENABLE_SCALAR_INT */
	#ifdef EDAMER_ENABLE_SCALAR_FLOAT
		EDAMER_TYPE_NAME_PAIRS(float),
		EDAMER_TYPE_NAME_PAIRS(float const),
	#endif /* EDAMER_ENABLE_SCALAR_FLOAT */
	#ifdef EDAMER_ENABLE_SCALAR_DOUBLE
		EDAMER_TYPE_NAME_PAIRS(double),
		EDAMER_TYPE_NAME_PAIRS(double const),
	#endif /* EDAMER_ENABLE_SCALAR_DOUBLE */
	#ifdef EDAMER_ENABLE_SCALAR_COMPLEX_FLOAT
		EDAMER_TYPE_NAME_PAIRS(El::Complex<float>),
	#endif /* EDAMER_ENABLE_SCALAR_COMPLEX_FLOAT */
	#ifdef EDAMER_ENABLE_SCALAR_COMPLEX_DOUBLE
		EDAMER_TYPE_NAME_PAIRS(El::Complex<double>),
	#endif /* EDAMER_ENABLE_SCALAR_COMPLEX_DOUBLE */
	"SEQUENCE_TERMINATOR___REMOVED_BY_DROP_BACK"
));

EDAMER_NAMESPACE_END(/* unnamed */)


#define _EL_VECTOR(vector_kind)                                                                                        \
	py::module &                                                                                                       \
	pydef_impl<hbrs::mpl::el_ ## vector_kind ## _vector_tag>::apply(py::module & m, py::module & base) {               \
		using hbrs::mpl::el_ ## vector_kind ## _vector;                                                                \
		using hbrs::mpl::el_ ## vector_kind ## _vector_tag;                                                            \
		auto py_el_vector =                                                                                            \
			py::class_<el_ ## vector_kind ## _vector_tag>{                                                             \
				m, pystrip("el_"#vector_kind"_vector").c_str()                                                         \
			};                                                                                                         \
		                                                                                                               \
		hana::for_each(                                                                                                \
			scalars,                                                                                                   \
			[&m, &py_el_vector](auto pairs) {                                                                          \
				auto types = hana::transform(pairs, hana::first);                                                      \
				auto names = hana::transform(pairs, hana::second);                                                     \
				auto name = boost::format("el_"#vector_kind"_vector<%s>") %                                            \
					hana::fold_left(names, [](std::string s, std::string name) { return s + ',' + name; });            \
				                                                                                                       \
				using ring_t = typename decltype(+hana::at_c<0>(types))::type;                                         \
				using type_t = el_ ## vector_kind ## _vector<ring_t>;                                                  \
				                                                                                                       \
				/* store template function pointers in variables to work around                                        \
				* "unresolved overloaded function type" errors with GCC9/10                                            \
				*/                                                                                                     \
				constexpr auto view_from_numpy_1d_ptr =                                                                \
					&view_from_numpy_1d_t<ring_t, el_ ## vector_kind ## _vector_tag>::apply;                           \
				constexpr auto view_to_numpy_1d_ptr =                                                                  \
					&view_to_numpy_1d_t<ring_t, el_ ## vector_kind ## _vector_tag>::apply;                             \
				constexpr auto length_ptr = &type_t::length;                                                           \
				                                                                                                       \
				py_el_vector.def_static("view_from_numpy",                                                             \
					view_from_numpy_1d_ptr,                                                                            \
					py::arg("array").noconvert(true),                                                                  \
					py::keep_alive<0, 1>());                                                                           \
				/* Previously, incompatible arrays were just accepted by mistake even if noconvert(true) [1], but since\
				 * pull request #2484 [2], pybind11 checks whether the array::c_style or array::f_style flags are      \
				 * satisifed and raises an error otherwise.                                                            \
				* Ref.:                                                                                                \
				* [1] https://github.com/pybind/pybind11/issues/2455                                                   \
				* [2] https://github.com/pybind/pybind11/pull/2484                                                     \
				*/                                                                                                     \
				                                                                                                       \
				py::class_<type_t>{m, pystrip(name.str()).c_str(), py_el_vector}                                       \
					.def(py::init<El::Int>())                                                                          \
					.def("length", length_ptr)                                                                         \
					.def("view_to_numpy", view_to_numpy_1d_ptr, py::keep_alive<0, 1>())                                \
					;                                                                                                  \
			}                                                                                                          \
		);                                                                                                             \
		                                                                                                               \
		return m;                                                                                                      \
	}

_EL_VECTOR(column)
_EL_VECTOR(row)
#undef _EL_VECTOR

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL