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
#include <hbrs/mpl/dt/matrix_index.hpp>
#include <hbrs/mpl/dt/matrix_size.hpp>
#include <hbrs/mpl/dt/range/impl.hpp>
#include <memory>
#include <pybind11/numpy.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

py::module &
pydef_impl<hbrs::mpl::range_tag>::apply(py::module & m, py::module & base) {
	using hbrs::mpl::range;
	using hbrs::mpl::range_tag;
	auto py_range = py::class_<range_tag>{m, pystrip("range").c_str()};
	
	using matrix_index_int_int = mpl::matrix_index<int,int>;
	using matrix_size_int_int = mpl::matrix_size<int,int>;
	
	hana::for_each(
		hana::make_tuple(
			EDAMER_TYPE_NAME_PAIRS(matrix_index_int_int, matrix_index_int_int), /* El::Int is a typedef for int */
			EDAMER_TYPE_NAME_PAIRS(matrix_index_int_int, matrix_size_int_int)
		),
		[&m, &py_range](auto pairs) {
			auto types = hana::transform(pairs, hana::first);
			auto names = hana::transform(pairs, hana::second);
			auto name = boost::format("range<%s>") %
				hana::fold_left(names, [](std::string s, std::string name) { return s + ',' + name; });
			
			using first_t = typename decltype(+hana::at_c<0>(types))::type;
			using last_t = typename decltype(+hana::at_c<1>(types))::type;
			using type_t = range<first_t, last_t>;
			
			/* store template function pointers in variables to work around
			 * "unresolved overloaded function type" errors with GCC9/10
			 */
			auto c = py::class_<type_t>{m, pystrip(name.str()).c_str(), py_range}
				.def(
					py::init<first_t, last_t>(),
					py::arg("first"),
					py::arg("last")
				);
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().first()), first_t&>) {
				c.def_property("first",
					[](type_t & rng) { return rng.first(); },
					[](type_t & rng, first_t & first) { rng.first() = HBRS_MPL_FWD(first); }
				);
			} else {
				c.def_property_readonly("first",
					[](type_t & rng) { return rng.first(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().last()), last_t&>) {
				c.def_property("last",
					[](type_t & rng) { return rng.last(); },
					[](type_t & rng, last_t & last) { rng.last() = HBRS_MPL_FWD(last); }
				);
			} else {
				c.def_property_readonly("last",
					[](type_t & rng) { return rng.last(); }
				);
			}
			
			py_range.def_static("make",
				[](first_t m_, last_t n_) -> type_t { return {m_, n_}; }
			);
		}
	);
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

