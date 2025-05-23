/*************************************************************************
 *   Copyright (c) 2025 - 2025 Yichao Yu <yyc1992@gmail.com>             *
 *                                                                       *
 *   This library is free software; you can redistribute it and/or       *
 *   modify it under the terms of the GNU Lesser General Public          *
 *   License as published by the Free Software Foundation; either        *
 *   version 3.0 of the License, or (at your option) any later version.  *
 *                                                                       *
 *   This library is distributed in the hope that it will be useful,     *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
 *   Lesser General Public License for more details.                     *
 *                                                                       *
 *   You should have received a copy of the GNU Lesser General Public    *
 *   License along with this library. If not,                            *
 *   see <http://www.gnu.org/licenses/>.                                 *
 *************************************************************************/

#include "src/utils.h"
#include "src/artiq_backend.h"
#include "src/artiq_utils.h"

using namespace brassboard_seq;

static PyModuleDef artiq_backend_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "brassboard_seq.artiq_backend",
    .m_size = -1,
};

PY_MODINIT(artiq_backend, artiq_backend_module)
{
    artiq_utils::patch_artiq();
    m.add_type(&artiq_backend::ArtiqBackend::Type);
}
