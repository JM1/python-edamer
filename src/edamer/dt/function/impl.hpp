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

#ifndef EDAMER_DT_FUNCTION_IMPL_HPP
#define EDAMER_DT_FUNCTION_IMPL_HPP

#include "fwd.hpp"

#define EDAMER_DEFINE_FUNCTION(f_type, f_name)                                                                         \
	PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)                                                                       \
	PYBIND11_NAMESPACE_BEGIN(detail)                                                                                   \
                                                                                                                       \
	bool                                                                                                               \
	type_caster<hbrs::mpl::f_type ## _t>::load(handle src, bool /* convert */) {                                       \
		if (!src) {                                                                                                    \
			return false;                                                                                              \
		}                                                                                                              \
		                                                                                                               \
		if (!src.is(module_::import("edamer").attr("cpp").attr("fn").attr(#f_type))) {                                 \
			return false;                                                                                              \
		}                                                                                                              \
		                                                                                                               \
		value = hbrs::mpl::f_type;                                                                                     \
		return true;                                                                                                   \
	}                                                                                                                  \
                                                                                                                       \
	handle                                                                                                             \
	type_caster<hbrs::mpl::f_type ## _t>::cast(                                                                        \
		hbrs::mpl::f_type ## _t const& src,                                                                            \
		return_value_policy /* policy */,                                                                              \
		handle /* parent */) {                                                                                         \
		return module_::import("edamer").attr("cpp").attr("fn").attr(#f_type);                                         \
	}                                                                                                                  \
                                                                                                                       \
	PYBIND11_NAMESPACE_END(detail)                                                                                     \
	PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#define EDAMER_DEF_F(...)                                                                                              \
	EDAMER_DEFINE_FUNCTION(__VA_ARGS__)

#endif // !EDAMER_DT_FUNCTION_IMPL_HPP
