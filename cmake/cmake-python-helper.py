#   Copyright (C) 2012~2013 by Yichao Yu
#   yyc1992@gmail.com
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, version 2 of the License.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the
#   Free Software Foundation, Inc.,
#   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

# This file incorporates work covered by the following copyright and
# permission notice:
#
#     Copyright (c) 2007, Simon Edwards <simon@simonzone.com>
#     Redistribution and use is allowed according to the terms of the BSD
#     license. For details see the accompanying COPYING-CMAKE-SCRIPTS file.

from __future__ import print_function


def get_sys_info():
    import sys
    print("exec_prefix:%s" % sys.exec_prefix)
    try:
        magic_tag = sys.implementation.cache_tag
    except:
        try:
            import imp
            magic_tag = imp.get_tag()
        except AttributeError:
            magic_tag = ''
    print("magic_tag:%s" % magic_tag)
    try:
        import Cython
        print('cython_version:%s' % Cython.__version__)
    except:
        print('cython_version:')
    import importlib
    print('extension_suffix:%s' % importlib.machinery.EXTENSION_SUFFIXES[0])
    return 0


def compile_file(infile):
    import py_compile
    try:
        py_compile.compile(infile, doraise=True)
        return 0
    except py_compile.PyCompileError as e:
        print(e.msg)
        return 1


def main(argv):
    if argv[1] == '--get-sys-info':
        return get_sys_info()
    elif argv[1] == '--compile':
        return compile_file(argv[2])
    else:
        import sys
        print('Unknown options %s' % argv[1:], file=sys.stderr)
        return 1

if '__main__' == __name__:
    import sys
    sys.exit(main(sys.argv))
