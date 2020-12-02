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

#include <boost/format.hpp>
#include <boost/hana/at.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <hbrs/mpl/dt/matrix_index/impl.hpp>
#include <memory>
#include <pybind11/numpy.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

py::module &
pydef_impl<hbrs::mpl::matrix_index_tag>::apply(py::module & m, py::module & base) {
	using hbrs::mpl::matrix_index;
	using hbrs::mpl::matrix_index_tag;
	auto py_matrix_index = py::class_<matrix_index_tag>{m, pystrip("matrix_index").c_str()};
	
	hana::for_each(
		hana::make_tuple(
			/* Both C++ types int and std::size_t will be mapped to Python's int type,
			 * hence dt.MatrixSize.make() will have multiple overloads:
			 *  1. make(arg0: int, arg1: int) -> edamer.cpp.dt.MatrixIndex_Int_Int
			 *  2. make(arg0: int, arg1: int) -> edamer.cpp.dt.MatrixIndex_StdSizeT_StdSizeT
			 * We list int before std::size_t to prioritize the MatrixIndex_Int_Int overload.
			 */
			EDAMER_TYPE_NAME_PAIRS(int, int), /* El::Int is a typedef for int */
			EDAMER_TYPE_NAME_PAIRS(std::size_t, std::size_t)
		),
		[&m, &py_matrix_index](auto pairs) {
			auto types = hana::transform(pairs, hana::first);
			auto names = hana::transform(pairs, hana::second);
			auto name = boost::format("matrix_index<%s>") %
				hana::fold_left(names, [](std::string s, std::string name) { return s + ',' + name; });
			
			using m_t = typename decltype(+hana::at_c<0>(types))::type;
			using n_t = typename decltype(+hana::at_c<1>(types))::type;
			using type_t = matrix_index<m_t, n_t>;
			
			/* store template function pointers in variables to work around
			 * "unresolved overloaded function type" errors with GCC9/10
			 */
			auto c = py::class_<type_t>{m, pystrip(name.str()).c_str(), py_matrix_index}
				.def(
					py::init<m_t, n_t>(),
					py::arg("m"),
					py::arg("n")
				);
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().m()), m_t&>) {
				c.def_property("m",
					[](type_t & idx) { return idx.m(); },
					[](type_t & idx, m_t & m) { idx.m() = HBRS_MPL_FWD(m); }
				);
			} else {
				c.def_property_readonly("m",
					[](type_t & idx) { return idx.m(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().n()), n_t&>) {
				c.def_property("n",
					[](type_t & idx) { return idx.n(); },
					[](type_t & idx, n_t & n) { idx.n() = HBRS_MPL_FWD(n); }
				);
			} else {
				c.def_property_readonly("n",
					[](type_t & idx) { return idx.n(); }
				);
			}
			
			py_matrix_index.def_static("make",
				[](m_t m_, n_t n_) -> type_t { return {m_, n_}; }
			);
		}
	);
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

