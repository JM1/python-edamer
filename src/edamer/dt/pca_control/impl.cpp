/* Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
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
#include <hbrs/mpl/dt/pca_control/impl.hpp>
#include <memory>
#include <pybind11/numpy.h>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

py::module &
pydef_impl<hbrs::mpl::pca_control_tag>::apply(py::module & m, py::module & base) {
	using hbrs::mpl::pca_control;
	using hbrs::mpl::pca_control_tag;

	auto py_pca_control = py::class_<pca_control_tag>{m, pystrip("pca_control").c_str()};
	
	hana::for_each(
		hana::make_tuple(
			EDAMER_TYPE_NAME_PAIRS(bool, bool, bool)
		),
		[&m, &py_pca_control](auto pairs) {
			auto types = hana::transform(pairs, hana::first);
			auto names = hana::transform(pairs, hana::second);
			auto name = boost::format("pca_control<%s>") %
				hana::fold_left(names, [](std::string s, std::string name) { return s + ',' + name; });
			
			using economy_t = typename decltype(+hana::at_c<0>(types))::type;
			using center_t = typename decltype(+hana::at_c<1>(types))::type;
			using normalize_t = typename decltype(+hana::at_c<2>(types))::type;
			
			using type_t = pca_control<economy_t, center_t, normalize_t>;
			
			/* store template function pointers in variables to work around
			 * "unresolved overloaded function type" errors with GCC9/10
			 */
			auto c = py::class_<type_t>{m, pystrip(name.str()).c_str(), py_pca_control}
				.def(
					py::init<economy_t, center_t, normalize_t>(),
					py::arg("economy"),
					py::arg("center"),
					py::arg("normalize")
				);
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().economy()), economy_t &>) {
				c.def_property("economy",
					[](type_t & o) { return o.economy(); },
					[](type_t & o, economy_t & v) { o.economy() = HBRS_MPL_FWD(v); }
				);
			} else {
				c.def_property_readonly("economy",
					[](type_t & o) { return o.economy(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().center()), center_t &>) {
				c.def_property("center",
					[](type_t & o) { return o.center(); },
					[](type_t & o, center_t & v) { o.center() = HBRS_MPL_FWD(v); }
				);
				
			} else {
				c.def_property_readonly("center",
					[](type_t & o) { return o.center(); }
				);
			}
			
			if constexpr (std::is_assignable_v<decltype(std::declval<type_t&>().normalize()), normalize_t &>) {
				c.def_property("normalize",
					[](type_t & o) { return o.normalize(); },
					[](type_t & o, normalize_t & v) { o.normalize() = HBRS_MPL_FWD(v); }
				);
			} else {
				c.def_property_readonly("normalize",
					[](type_t & o) { return o.normalize(); }
				);
			}
			
			py_pca_control.def_static("make",
				[](economy_t economy, center_t center, normalize_t normalize) {
					return type_t{economy, center, normalize};
				},
				py::arg("economy"),
				py::arg("center"),
				py::arg("normalize")
			);
		}
	);
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

