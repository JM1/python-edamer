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

#include <hbrs/mpl/dt/exception.hpp>

#define _REGISTER_EXCEPTION(m, e, base)                                                                                \
	py::register_exception<e>(m, pystrip(#e).c_str(), base)

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace mpl = hbrs::mpl;

py::module &
pydef_impl<exception_tag>::apply(py::module & m, py::module & base) {
	using mpl::exception;
	using mpl::incompatible_matrix_exception;
	auto ex = _REGISTER_EXCEPTION(m, exception, PyExc_Exception);
	_REGISTER_EXCEPTION(m, incompatible_matrix_exception, ex);
	_REGISTER_EXCEPTION(m, incompatible_ndarray_exception, ex);
	_REGISTER_EXCEPTION(m, import_mpi4py_failed_exception, ex);
	_REGISTER_EXCEPTION(m, matrix_distribution_not_supported_exception,ex);
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)
