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

#include <hbrs/mpl/dt/expression/impl.hpp>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

py::module &
pydef_impl<hbrs::mpl::expression_tag>::apply(py::module & m, py::module & base) {
	using hbrs::mpl::expression_tag;
	
	auto py_expression = py::class_<expression_tag>{m, pystrip("expression").c_str()};
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)
