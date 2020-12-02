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

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include <boost/format.hpp>
#include <boost/hana/at.hpp>
#include <boost/hana/concat.hpp>
#include <boost/hana/first.hpp>
#include <boost/hana/fold_left.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/transform.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/type.hpp>
#include <boost/hana/second.hpp>
#include <boost/hana/zip.hpp>
#include <boost/throw_exception.hpp>
#include <edamer/detail/scalar.hpp>
#include <edamer/dt/el_complex.hpp>
#include <edamer/dt/exception.hpp>
#include <edamer/dt/matrix_distribution.hpp>
#include <hbrs/mpl/dt/el_dist_matrix/impl.hpp>
#include <hbrs/mpl/dt/el_matrix.hpp>
#include <hbrs/mpl/dt/matrix_distribution.hpp>
#include <pybind11/numpy.h>
#include <tuple>
#include <unordered_map>

EDAMER_NAMESPACE_BEGIN(EDAMER_NAMESPACE)
namespace hana = boost::hana;
namespace mpl = hbrs::mpl;

EDAMER_NAMESPACE_BEGIN(/* unnamed */)

template<
	typename Ring,
	El::Dist Columnwise,
	El::Dist Rowwise,
	El::DistWrap Wrapping
>
py::object
make_view(
	El::Grid const& grid,
	mpl::el_matrix<Ring> & local,
	mpl::matrix_distribution<
		hana::integral_constant<El::Dist, Columnwise>,
		hana::integral_constant<El::Dist, Rowwise>,
		hana::integral_constant<El::DistWrap, Wrapping>
	>
) {
	using Ring_no_Ref = std::remove_const_t<Ring>;
	El::DistMatrix<Ring_no_Ref, Columnwise, Rowwise, Wrapping> global_el {grid};
	
	if constexpr(std::is_const_v<Ring>) {
		global_el.LockedAttach(
			local.data().Height(), local.data().Width(), grid, 0, 0, local.data().LockedBuffer(), local.data().LDim());
		return py::cast(mpl::make_el_dist_matrix(std::move(global_el)));
	} else {
		global_el.Attach(
			local.data().Height(), local.data().Width(), grid, 0, 0, local.data().Buffer(), local.data().LDim());
		return py::cast(mpl::make_el_dist_matrix(std::move(global_el)));
	}
}

template<
	typename Ring,
	El::Dist Columnwise,
	El::Dist Rowwise,
	El::DistWrap Wrapping
>
auto
local(py::object &obj) {
	using matrix_t = mpl::el_dist_matrix<Ring, Columnwise, Rowwise, Wrapping>;
	matrix_t & matrix = obj.cast<matrix_t&>();
	
	if constexpr(std::is_const_v<Ring>) {
		decltype(auto) local = matrix.data().LockedMatrix();
		return mpl::el_matrix<Ring>{El::Matrix<Ring>{local.Height(), local.Width(), local.LockedBuffer(), local.LDim()}};
	} else {
		decltype(auto) local = matrix.data().Matrix();
		return mpl::el_matrix<Ring>{El::Matrix<Ring>{local.Height(), local.Width(), local.Buffer(), local.LDim()}};
	}
}

template<
	typename Ring,
	El::Dist FromColumnwise,
	El::Dist FromRowwise,
	El::DistWrap FromWrapping,
	El::Dist ToColumnwise = FromColumnwise,
	El::Dist ToRowwise = FromRowwise,
	El::DistWrap ToWrapping = FromWrapping
>
decltype(auto)
copy(
	mpl::el_dist_matrix<Ring, FromColumnwise, FromRowwise, FromWrapping> const& from,
	mpl::matrix_distribution<
		hana::integral_constant<El::Dist, ToColumnwise>,
		hana::integral_constant<El::Dist, ToRowwise>,
		hana::integral_constant<El::DistWrap, ToWrapping>
	> const& to_dist = {
		hana::integral_constant<El::Dist, FromColumnwise>{},
		hana::integral_constant<El::Dist, FromRowwise>{},
		hana::integral_constant<El::DistWrap, FromWrapping>{}
	}
) {
	return mpl::make_el_dist_matrix(from, to_dist);
}

EDAMER_NAMESPACE_END(/* unnamed */)

py::module &
pydef_impl<hbrs::mpl::el_dist_matrix_tag>::apply(py::module & m, py::module & base) {
	// El::DistMatrix<> does not support const-qualified or unsigned or long types
	// Ref.: https://github.com/elemental/Elemental/blob/6eb15a0da2a4998bf1cf971ae231b78e06d989d9/src/core/DistMatrix/Element/MC_MR.cpp#L284
	auto ring_tns = hana::concat(detail::scalars, detail::complex_scalars);
	
	using hbrs::mpl::el_matrix;
	using hbrs::mpl::el_dist_matrix;
	using hbrs::mpl::el_dist_matrix_tag;
	auto py_el_dist_matrix = py::class_<el_dist_matrix_tag>{m, pystrip("el_dist_matrix").c_str()};
	
	hana::for_each(ring_tns, [&m, &py_el_dist_matrix](auto ring_tn) {
		using ring_t = typename decltype(+hana::first(ring_tn))::type;
		auto ring_n = hana::second(ring_tn);
		
		/* Make class_<> objects known to pybind11 first and add their methods later. This is required to handle
		 * circular dependencies between distributed matrices, e.g. it will enable Python's help() function to print the
		 * Python class names instead of printing the C++ class names for the copy() member function.
		 */
		auto py_el_dist_matrix_insts = hana::transform(el_matrix_distributions, [&](auto distribution_tn) {
			auto dist_ts = hana::transform(distribution_tn, hana::first);
			auto dist_ns = hana::transform(distribution_tn, hana::second);
			
			auto name = boost::format("el_dist_matrix<%s,%s>") % ring_n %
				hana::fold_left(dist_ns, [](std::string s, std::string name) { return s + ',' + name; });
				
			using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;
			using rowwise_t = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;
			using wrapping_t = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;
			
			using dist_matrix_t = el_dist_matrix<ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value>;
			
			return py::class_<dist_matrix_t>{m, pystrip(name.str()).c_str(), py_el_dist_matrix};
		});
		
		// Now add class_<> functionality, e.g. member functions
		hana::for_each(hana::zip(el_matrix_distributions, py_el_dist_matrix_insts), [&](auto zipped) {
			auto distribution_tn = hana::at_c<0>(zipped);
			auto py_el_dist_matrix_inst = hana::at_c<1>(zipped);
				
			auto dist_ts = hana::transform(distribution_tn, hana::first);
			
			using columnwise_t = std::decay_t<decltype(hana::at_c<0>(dist_ts))>;
			using rowwise_t = std::decay_t<decltype(hana::at_c<1>(dist_ts))>;
			using wrapping_t = std::decay_t<decltype(hana::at_c<2>(dist_ts))>;
			
			using dist_matrix_t = el_dist_matrix<ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value>;
			
			/* store template function pointers in variables to work around
			* "unresolved overloaded function type" errors with GCC9/10
			*/
			constexpr auto make_view_ptr = &make_view<ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value>;
			constexpr auto size_ptr = &dist_matrix_t::size;
			constexpr auto local_ptr = &local<ring_t, columnwise_t::value, rowwise_t::value, wrapping_t::value>;
			constexpr auto participating_ptr = &dist_matrix_t::participating;
			
			py_el_dist_matrix_inst
				.def(py::init<El::Grid const&, El::Int, El::Int>(), py::keep_alive<1, 2>())
				.def("size", size_ptr)
				.def("local", local_ptr, py::keep_alive<0, 1>())
				.def("participating", participating_ptr, "Return True if this process can be assigned matrix data")
				;
			
			py_el_dist_matrix.def_static("make_view",
				py::overload_cast<
					El::Grid const&,
					el_matrix<ring_t> &,
					mpl::matrix_distribution<
						hana::integral_constant<El::Dist, columnwise_t::value>,
						hana::integral_constant<El::Dist, rowwise_t::value>,
						hana::integral_constant<El::DistWrap, wrapping_t::value>
					>
				>(make_view_ptr),
				py::keep_alive<0, 1>(),
				py::keep_alive<0, 2>()
			);
			
			hana::for_each(el_matrix_distributions, [&](auto to_distribution_tn) {
				using from_columnwise_t = columnwise_t;
				using from_rowwise_t = rowwise_t;
				using from_wrapping_t = wrapping_t;
				auto to_dist_ts = hana::transform(to_distribution_tn, hana::first);
				using to_columnwise_t = std::decay_t<decltype(hana::at_c<0>(to_dist_ts))>;
				using to_rowwise_t = std::decay_t<decltype(hana::at_c<1>(to_dist_ts))>;
				using to_wrapping_t = std::decay_t<decltype(hana::at_c<2>(to_dist_ts))>;
				
				constexpr auto copy_ptr = &copy<
					ring_t,
					from_columnwise_t::value, from_rowwise_t::value, from_wrapping_t::value,
					to_columnwise_t::value, to_rowwise_t::value, to_wrapping_t::value
				>;

				if /* constexpr // but does not compile with GCC9/10 */ (
					std::is_same_v<from_columnwise_t, to_columnwise_t> &&
					std::is_same_v<from_rowwise_t, to_rowwise_t> &&
					std::is_same_v<from_wrapping_t, to_wrapping_t>) {
					// Allow to call copy() without an argument to create a copy with the same matrix distribution
					py_el_dist_matrix_inst.def("copy",
						py::overload_cast<
							mpl::el_dist_matrix<
								ring_t,
								from_columnwise_t::value,
								from_rowwise_t::value,
								from_wrapping_t::value
							> const&,
							mpl::matrix_distribution<
								hana::integral_constant<El::Dist, to_columnwise_t::value>,
								hana::integral_constant<El::Dist, to_rowwise_t::value>,
								hana::integral_constant<El::DistWrap, to_wrapping_t::value>
							> const&
						>(copy_ptr),
						py::arg("to_dist") = mpl::make_matrix_distribution(
							hana::integral_constant<El::Dist, to_columnwise_t::value>{},
							hana::integral_constant<El::Dist, to_rowwise_t::value>{},
							hana::integral_constant<El::DistWrap, to_wrapping_t::value>{}
						)
					);
				} else {
					py_el_dist_matrix_inst.def("copy",
						py::overload_cast<
							mpl::el_dist_matrix<
								ring_t,
								from_columnwise_t::value,
								from_rowwise_t::value,
								from_wrapping_t::value
							> const&,
							mpl::matrix_distribution<
								hana::integral_constant<El::Dist, to_columnwise_t::value>,
								hana::integral_constant<El::Dist, to_rowwise_t::value>,
								hana::integral_constant<El::DistWrap, to_wrapping_t::value>
							> const&
						>(copy_ptr),
						py::arg("to_dist")
					);
				}
			});
		});
	});
	return m;
}

EDAMER_NAMESPACE_END(EDAMER_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL
