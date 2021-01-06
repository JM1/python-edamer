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

#ifndef EDAMER_DT_EL_COMPLEX_IMPL_HPP
#define EDAMER_DT_EL_COMPLEX_IMPL_HPP

#include "fwd.hpp"

#include <hbrs/mpl/config.hpp>

#ifdef HBRS_MPL_ENABLE_ELEMENTAL

#include <El.hpp>

// NOTE: Keep in sync with <pybind11/numpy.h>
#include <pybind11/numpy.h>
PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
template <typename T> struct is_complex<El::Complex<T>> : std::true_type { };
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

// The following code has been copied from <pybind11/numpy.h> and then std::complex has been replaced with El::Complex
// NOTE: Keep in sync with <pybind11/complex.h>
#include <pybind11/complex.h>
PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

template <typename T> struct format_descriptor<El::Complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr const char c = format_descriptor<T>::c;
    static constexpr const char value[3] = { 'Z', c, '\0' };
    static std::string format() { return std::string(value); }
};

#ifndef PYBIND11_CPP17

template <typename T> constexpr const char format_descriptor<
    El::Complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>>::value[3];

#endif

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T> struct is_fmt_numeric<El::Complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr bool value = true;
    static constexpr int index = is_fmt_numeric<T>::index + 3;
};

template <typename T> class type_caster<El::Complex<T>> {
public:
    bool load(handle src, bool convert) {
        if (!src)
            return false;
        if (!convert && !PyComplex_Check(src.ptr()))
            return false;
        Py_complex result = PyComplex_AsCComplex(src.ptr());
        if (result.real == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        value = El::Complex<T>((T) result.real, (T) result.imag);
        return true;
    }

    static handle cast(const El::Complex<T> &src, return_value_policy /* policy */, handle /* parent */) {
        return PyComplex_FromDoubles((double) src.real(), (double) src.imag());
    }

    PYBIND11_TYPE_CASTER(El::Complex<T>, _("complex"));
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif // HBRS_MPL_ENABLE_ELEMENTAL
#endif // !EDAMER_DT_EL_COMPLEX_IMPL_HPP
