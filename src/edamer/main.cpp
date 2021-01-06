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

#include <boost/hana/first.hpp>
#include <boost/hana/flatten.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/pair.hpp>
#include <boost/hana/second.hpp>
#include <edamer/config.hpp>
#include <edamer/detail/log.hpp>
#include <edamer/detail/pybind11.hpp>
#include <edamer/detail/scalar.hpp>
#include <edamer/detail/test.hpp>
#include <edamer/dt/el_dist_matrix.hpp>
#include <edamer/dt/el_dist_vector.hpp>
#include <edamer/dt/el_grid.hpp>
#include <edamer/dt/el_matrix.hpp>
#include <edamer/dt/el_vector.hpp>
#include <edamer/dt/exception.hpp>
#include <edamer/dt/expression.hpp>
#include <edamer/dt/matrix_distribution.hpp>
#include <edamer/dt/matrix_index.hpp>
#include <edamer/dt/matrix_size.hpp>
#include <edamer/dt/pca_control.hpp>
#include <edamer/dt/pca_result.hpp>
#include <edamer/dt/range.hpp>
#include <edamer/fn/expand.hpp>
#include <edamer/fn/multiply.hpp>
#include <edamer/fn/pca.hpp>
#include <edamer/fn/plus.hpp>
#include <edamer/fn/select.hpp>
#include <edamer/fn/size.hpp>
#include <edamer/fn/transpose.hpp>
#include <hbrs/mpl/detail/environment.hpp>

hbrs::mpl::detail::environment mpl_env{}; // Required e.g. for MPI initialization

namespace hana = boost::hana;
namespace py = pybind11;

PYBIND11_MODULE(cpp, m) {
	m.doc() = "EDAMER: Exascale Data Analysis Methods with Enhanced Reusability";
	
	py::module m_core = m.def_submodule("core", "Fundamentals");
	py::module m_detail = m.def_submodule("detail", "Internals");
	py::module m_dt = m.def_submodule("dt", "Datatypes");
	py::module m_fn = m.def_submodule("fn", "Functions");

	hana::for_each(
		hana::make_tuple(
			/* NOTE: Order of Python module initialization must follow the order of data type dependencies, e.g. the
			 *       pybind11 module code for distributed Elemental matrices must be executed after the module code for
			 *       Elemental's non-distributed matrices and Grid data structures. Else Python's help system will show
			 *       e.g. the C++ type name El::Grid instead of the mapped pybind11 class name edamer.dt.ElGrid. It does
			 *       not effect functionality negatively though. */
			hana::pair(m_core, hana::flatten(hana::make_tuple(
				/*, ...*/
			))),
			hana::pair(m_detail, hana::flatten(hana::make_tuple(
				EDAMER_DETAIL_PYBIND11_PYDEFS,
				EDAMER_DETAIL_LOG_PYDEFS,
				EDAMER_DETAIL_SCALAR_PYDEFS,
                EDAMER_DETAIL_TEST_PYDEFS /*, ...*/
			))),
			hana::pair(m_dt, hana::flatten(hana::make_tuple(
				EDAMER_DT_MATRIX_INDEX_PYDEFS,
				EDAMER_DT_MATRIX_SIZE_PYDEFS,
				EDAMER_DT_RANGE_PYDEFS,
				EDAMER_DT_MATRIX_DISTRIBUTION_PYDEFS,
				EDAMER_DT_EXCEPTION_PYDEFS,
				EDAMER_DT_EL_MATRIX_PYDEFS,
				EDAMER_DT_EL_VECTOR_PYDEFS,
				EDAMER_DT_EL_GRID_PYDEFS,
				EDAMER_DT_EL_DIST_MATRIX_PYDEFS,
				EDAMER_DT_EL_DIST_VECTOR_PYDEFS,
				EDAMER_DT_EXPRESSION_PYDEFS,
				EDAMER_DT_PCA_CONTROL_PYDEFS,
				EDAMER_DT_PCA_RESULT_PYDEFS /*, ...*/
			))),
			hana::pair(m_fn, hana::flatten(hana::make_tuple(
				EDAMER_FN_EXPAND_PYDEFS,
				EDAMER_FN_MULTIPLY_PYDEFS,
				EDAMER_FN_PCA_PYDEFS,
				EDAMER_FN_PLUS_PYDEFS,
				EDAMER_FN_SELECT_PYDEFS,
				EDAMER_FN_SIZE_PYDEFS,
				EDAMER_FN_TRANSPOSE_PYDEFS /*, ...*/
			)))
		),
		[&m](auto && m_and_pydefs) {
			auto & subm = hana::first(m_and_pydefs);
			auto pydefs = hana::second(m_and_pydefs);
			hana::for_each(
				pydefs,
				[&m, &subm](auto && pydef) {
					HBRS_MPL_FWD(pydef)(subm, m);
				}
			);
		}
	);
}
