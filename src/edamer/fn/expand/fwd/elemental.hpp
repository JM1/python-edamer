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

#ifndef EDAMER_FN_EXPAND_FWD_ELEMENTAL_HPP
#define EDAMER_FN_EXPAND_FWD_ELEMENTAL_HPP

#include <boost/hana/tuple.hpp>
#include <edamer/config.hpp>
#include <edamer/detail/pybind11.hpp>
#include <hbrs/mpl/config.hpp>
#include <hbrs/mpl/fn/expand/fwd.hpp>

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

template <>
struct pydef_impl<hbrs::mpl::detail::expand_impl_el_row_vector>;

template <>
struct pydef_impl<hbrs::mpl::detail::expand_impl_el_column_vector>;

template <>
struct pydef_impl<hbrs::mpl::detail::expand_impl_el_dist_row_vector>;

template <>
struct pydef_impl<hbrs::mpl::detail::expand_impl_el_dist_column_vector>;

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#define EDAMER_FN_EXPAND_PYDEFS_ELEMENTAL boost::hana::make_tuple(                                                     \
		edamer::pydef<hbrs::mpl::detail::expand_impl_el_row_vector>,                                                   \
		edamer::pydef<hbrs::mpl::detail::expand_impl_el_column_vector>,                                                \
		edamer::pydef<hbrs::mpl::detail::expand_impl_el_dist_row_vector>,                                              \
		edamer::pydef<hbrs::mpl::detail::expand_impl_el_dist_column_vector>                                            \
	)

#else // !HBRS_MPL_ENABLE_ELEMENTAL
#define EDAMER_FN_EXPAND_PYDEFS_ELEMENTAL boost::hana::make_tuple()
#endif // !HBRS_MPL_ENABLE_ELEMENTAL

#endif // !EDAMER_FN_EXPAND_FWD_ELEMENTAL_HPP
