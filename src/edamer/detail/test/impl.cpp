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


#include <hbrs/mpl/detail/test.hpp>
#include <pybind11/numpy.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

EDAMER_NAMESPACE_BEGIN(/* unnamed */)

template<typename T>
auto
c_array_view_to_numpy(T * data, std::size_t m, std::size_t n) {
	auto shape = py::array::ShapeContainer{m, n};
	auto strides = py::array::StridesContainer{ sizeof(T) * n, sizeof(T) };
	void * buf_ptr;
	if constexpr (std::is_const_v<std::remove_reference_t<T>>) {
		buf_ptr = const_cast<void*>(static_cast<void const*>(data));
		/* is safe because it will be casted to const void* in py::array_t */
	} else {
		buf_ptr = static_cast<void*>(data);
	}
	
	auto buf = py::buffer_info(
		buf_ptr,                                          /* Pointer to buffer */
		sizeof(T),                                        /* Size of one scalar */
		py::format_descriptor<std::decay_t<T>>::format(), /* Python struct-style format descriptor */
		2,                                                /* Number of dimensions */
		shape,                                            /* Buffer dimensions */
		strides,                                          /* Strides (in bytes) for each index */
		data                                              /* Buffer is readonly */
	);
	
	auto && array = py::array_t<std::decay_t<T>, py::array::c_style>{buf};
	
	if constexpr (std::is_const_v<std::remove_reference_t<T>>) {
		// Uses pybind11's private API to remove writeable flag because there is currently no official API for that
		// Ref.: https://github.com/pybind/pybind11/issues/481
		//TODO: Replace private API usage once it's supported!
		py::detail::array_proxy(array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
	}
	
	return HBRS_MPL_FWD(array);
}

struct test_db{};

EDAMER_NAMESPACE_END(/* unnamed */)

py::module &
pydef_impl<test_tag>::apply(py::module & m, py::module & base) {
	auto py_test_db = py::class_<test_db>{m, "TestDB"};

	#define _MATRIX(nr)                                                                                                \
		py_test_db.def_property_readonly_static(pystrip(std::string{"matrix_"} + #nr).c_str(),                         \
			[](py::object /* self */) {                                                                                \
				return c_array_view_to_numpy(                                                                          \
					mpl::detail::mat_ ## nr,                                                                           \
					mpl::detail::mat_ ## nr ## _m,                                                                     \
					mpl::detail::mat_ ## nr ## _n                                                                      \
				);                                                                                                     \
			});

	_MATRIX(a)
	_MATRIX(b)
	_MATRIX(c)
	_MATRIX(d)
	_MATRIX(e)
	_MATRIX(f)
	_MATRIX(g)
	_MATRIX(h)
	_MATRIX(i)
	_MATRIX(j)
	_MATRIX(k)
	_MATRIX(l)
	_MATRIX(m)
	_MATRIX(n)
	_MATRIX(o)
	_MATRIX(p)
	
	#undef _MATRIX
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)
