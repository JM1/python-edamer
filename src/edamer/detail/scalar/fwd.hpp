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

#ifndef EDAMER_DETAIL_SCALAR_FWD_HPP
#define EDAMER_DETAIL_SCALAR_FWD_HPP

#include <boost/hana/tuple.hpp>
#include <edamer/config.hpp>
#include <edamer/detail/pybind11.hpp>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

struct scalar_tag{};

template <>
struct pydef_impl<scalar_tag>;

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#define EDAMER_DETAIL_SCALAR_PYDEFS boost::hana::make_tuple(                                                           \
		edamer::pydef<edamer::scalar_tag>                                                                              \
	)

#endif // !EDAMER_DETAIL_SCALAR_FWD_HPP
