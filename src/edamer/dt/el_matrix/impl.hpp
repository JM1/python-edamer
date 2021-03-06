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

#ifndef EDAMER_DT_EL_MATRIX_IMPL_HPP
#define EDAMER_DT_EL_MATRIX_IMPL_HPP

#include "fwd.hpp"

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace mpl = hbrs::mpl;

template <>
struct EDAMER_API pydef_impl<hbrs::mpl::el_matrix_tag> {
	static py::module &
	apply(py::module & m, py::module & base);
};

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL
#endif // !EDAMER_DT_EL_MATRIX_IMPL_HPP
