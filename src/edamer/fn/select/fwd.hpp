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

#ifndef EDAMER_FN_SELECT_FWD_HPP
#define EDAMER_FN_SELECT_FWD_HPP

#include <boost/hana/flatten.hpp>
#include <edamer/config.hpp>
#include <edamer/dt/function/fwd.hpp>
#include <hbrs/mpl/fn/select/fwd.hpp>

#include "fwd/elemental.hpp"

EDAMER_DEC_F(select, "FnSelect")

#define EDAMER_FN_SELECT_PYDEFS boost::hana::flatten(boost::hana::make_tuple(                                          \
		EDAMER_FN_SELECT_PYDEFS_ELEMENTAL                                                                              \
	))

#endif // !EDAMER_FN_SELECT_FWD_HPP
