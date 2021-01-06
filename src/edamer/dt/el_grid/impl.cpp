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

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include <boost/assert.hpp>
#include <boost/throw_exception.hpp>
#include <edamer/dt/exception.hpp>
#include <El.hpp>
#include <mpi4py/mpi4py.h>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

bool
type_caster<El::mpi::Comm>::load(handle src, bool) {
	if (!src) {
		return false;
	}
	
	if(!PyObject_TypeCheck(src.ptr(), &PyMPIComm_Type)) {
		return false;
	}
	
	auto comm_ptr = reinterpret_cast<PyMPICommObject*>(src.ptr());
	BOOST_ASSERT(comm_ptr != nullptr);

	static_assert(std::is_same_v<MPI_Comm, std::decay_t<decltype(comm_ptr->ob_mpi)>>, "unexcepted PyMPIComm_Type pointer");
	value = El::mpi::Comm{comm_ptr->ob_mpi};
	return true;
}

handle
type_caster<El::mpi::Comm>::cast(El::mpi::Comm const& src, return_value_policy /* policy */, handle /* parent */) {
	BOOST_ASSERT(PyMPIComm_New != nullptr);
	return PyMPIComm_New(src.comm);
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)

py::module &
pydef_impl<edamer::el_grid_tag>::apply(py::module & m, py::module & base) {
	/* NOTE: Function import_mpi4py() will load mpi4py's Python module, which then will call MPI_Init() if MPI has not
	 * yet been initialized [1]. If mpi4py did this call to MPI_Init(), then it will call MPI_Finalize() once the Python
	 * module is unloaded. Elemental's El::Environment class [2], which is loaded e.g. in hbrs::mpl's environment
	 * class [3], will call MPI_Init() in its constructor and MPI_Finalize() in its destructor regardless of whether or
	 * not MPI has been initialized before. If mpi4py finalizes MPI before Elemental has called its Finalize()
	 * function [4], then the whole MPI job will crash with this error message:
	 *   "Warning: MPI was finalized before Elemental.
	 *    *** The MPI_Op_free() function was called after MPI_FINALIZE was invoked.
	 *    *** This is disallowed by the MPI standard.
	 *    *** Your MPI job will now abort.
	 *    [2a0a7532f32c:01074] Local abort after MPI_FINALIZE started completed successfully, but am not able to
	 *    aggregate error messages, and not able to guarantee that all other processes were killed!"
	 * To prevent this, we must ensure that MPI is initialized by Elemental. Hence the call to import_mpi4py() must
	 * happen after Elemental's Environment [2] or hbrs::mpl's environment [3] classes have been instantiated. This
	 * forbids calls to import_mpi4py() in a static class constructor, because the static variable initialization order
	 * is hard to get right [5].
	 *
	 * Ref.:
	 * [1] https://github.com/mpi4py/mpi4py/blob/9da9110e162de3740e87cb157f3acb3c0872ec69/src/mpi4py/MPI/atimport.pxi#L150
	 * [2] https://github.com/elemental/Elemental/blob/6eb15a0da2a4998bf1cf971ae231b78e06d989d9/include/El/core/environment/decl.hpp#L55
	 * [3] https://github.com/JM1/hbrs-mpl/blob/58ebe65230210b74a1c3ef7c0e52aada2a932739/src/hbrs/mpl/detail/environment/impl.cpp#L113
	 * [4] https://github.com/elemental/Elemental/blob/6eb15a0da2a4998bf1cf971ae231b78e06d989d9/src/core/environment.cpp#L200
	 * [5] https://stackoverflow.com/a/211307/6490710
	 */
	/* NOTE: mpi4py's global variables are static, i.e. they have internal linkage, and hence cannot be accessed by name
	 * in another translation unit. Therefore, in order to call mpi4py's initialize function import_mpi4py(), the
	 * environment variable has to be declared in each translation unit separately!
	 */
	// Required to use mpi4py's functions and variables
	if (import_mpi4py() < 0) {
		BOOST_THROW_EXCEPTION(import_mpi4py_failed_exception{});
	}
	
	py::enum_<El::GridOrder>(m, pystrip("El::GridOrder").c_str())
		.value("ROW_MAJOR", El::ROW_MAJOR)
		.value("COLUMN_MAJOR", El::COLUMN_MAJOR)
		.export_values();
	
	py::class_<El::Grid>{m, pystrip("El::Grid").c_str()}
		.def(py::init<El::mpi::Comm>())
// 		.def(py::init())
		.def("comm", &El::Grid::Comm);
	
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL