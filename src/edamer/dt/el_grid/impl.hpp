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

#ifndef EDAMER_DT_EL_GRID_IMPL_HPP
#define EDAMER_DT_EL_GRID_IMPL_HPP

#include "fwd.hpp"

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include <El.hpp>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <>
class EDAMER_API type_caster<El::mpi::Comm> {
public:
	bool
	load(handle src, bool);
	
	static handle
	cast(El::mpi::Comm const& src, return_value_policy /* policy */, handle /* parent */);
	
	PYBIND11_TYPE_CASTER(El::mpi::Comm, _("mpi4py.MPI.Comm"));
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

template <>
struct EDAMER_API pydef_impl<edamer::el_grid_tag> {
	static py::module &
	apply(py::module & m, py::module & base);
};

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL
#endif // !EDAMER_DT_EL_GRID_IMPL_HPP
