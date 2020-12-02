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

#include <boost/hana/first.hpp>
#include <boost/hana/for_each.hpp>
#include <edamer/detail/scalar.hpp>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace mpl = hbrs::mpl;

EDAMER_NAMESPACE_BEGIN(/* unnamed */)

py::list
scalars() {
	py::list list;
	hana::for_each(
		detail::scalars,
		[&list](auto pair) {
			using scalar_t = typename decltype(+hana::first(pair))::type;
			list.append(py::dtype::of<scalar_t>());
		}
	);
    return list;
}

py::list
complex_scalars() {
	py::list list;
	hana::for_each(
		detail::complex_scalars,
		[&list](auto pair) {
			using scalar_t = typename decltype(+hana::first(pair))::type;
			list.append(py::dtype::of<scalar_t>());
		}
	);
    return list;
}

EDAMER_NAMESPACE_END(/* unnamed */)

py::module &
pydef_impl<edamer::scalar_tag>::apply(py::module & m, py::module & base) {
	m.def("scalars", &scalars);
	m.def("complex_scalars", &complex_scalars);
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)
