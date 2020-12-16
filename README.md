# Python library `edamer` feat. distributed algorithms powered by [`hbrs-mpl`][hbrs-mpl] and [`Elemental`][elemental]

`edamer` is a Python 3 library
([GitHub.com][py-edamer], [H-BRS GitLab][hbrs-gitlab-py-edamer], [FHG GitLab][fhg-gitlab-py-edamer])
that provides data structures and algorithms like PCA for distributed scientific computing at HPC clusters.
Our research goal is to make the full power of generic C++ libraries usable for data analysis
and machine learning applications in Python without sacrificing space and time efficiency.
For example, `edamer` allows to run a distributed PCA on CFD data from [NEK5000][nek5000] in-situ from a
[ParaView Catalyst][pv-catalyst-guide] script written in Python only.

Its development started in Juli 2020 as part of the [`EDAMER`][fhg-gitlab-edamer] research project and is funded by
[Fraunhofer SCAI][fhg-scai]. `EDAMER` is an acronym for `Exascale Data Analysis Methods with Enhanced Reusability` and
expresses our ambition to bring our efficient and scalable software components for dense linear algebra and dimension
reduction from C++ library [`hbrs-mpl`][hbrs-mpl] to the broader audience of Python developers. We want to equip
scientists with a generic framework and a predefined set of reusable and robust components which allows them to codify
complex algorithms for exascale computers rapidly, without need for in-depth programming knowledge in C++ or HPC.
Numerical analysis of large simulation datasets is a representative domain for generic libraries, as quick and easy
comparison of varied algorithms is of particular interest here.

:warning: **WARNING:**
This code is still under development and is not ready for production yet.
Code might change or break at any time and is missing proper documentation.
:warning:

## Example: Distributed PCA in 12 lines of Python code

```Python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Jakob Meng, <jakobmeng@web.de>

from edamer import detail, dt, fn
from mpi4py import MPI
import numpy as np

# Generate example data
dataset = np.asarray(np.arange(1000*2000).reshape(1000, 2000), order='F')

# Wrap NumPy array in Elemental matrix (no copy)
matrix = dt.ElMatrix.view_from_numpy(dataset)

# Construct distributed Elemental matrix from local matrices (no copy)
grid = dt.ElGrid(MPI.COMM_WORLD)
star_star = dt.MatrixDistribution.make(dt.ElDist.STAR, dt.ElDist.STAR, dt.ElDistWrap.ELEMENT)
dist_matrix = dt.ElDistMatrix.make_view(grid, matrix, star_star)

# Apply PCA on distributed Elemental matrix
pca_ctrl = dt.PcaControl.make(economy=True, center=False, normalize=False)
dec = fn.pca(dist_matrix, pca_ctrl)

# Rebuild and test dataset
rebuild = fn.multiply(dec.score, fn.transpose(dec.coeff))
assert detail.test.matrix_matrix_allclose(dataset, rebuild)
```

## Under the hood

`edamer`'s [functions][py-edamer-fn] are mostly geared towards compatibility with [MATLAB's API][matlab-help], because
the latter has a strong focus on mathematical notations, is properly documented and useful for rapid prototyping.

`edamer` builds heavily upon C++ libraries [`hbrs-mpl`][hbrs-mpl] and [`Elemental`][elemental] which provide HPC-ready
data structures and algorithms for linear algebra and dimension reduction.

The full tech stack consists of:
* [`Python 3`][python3-ref] for user code
* [`C++17`][cpp-ref] for generic and efficient library code
* [`pybind11`][pybind11-doc] for language interop between Python and C++
* C++ library [`hbrs-mpl`][hbrs-mpl] ([GitHub.com][hbrs-mpl], [H-BRS GitLab][hbrs-gitlab-hbrs-mpl])
* C++ library [`Elemental`][elemental]
* C++ metaprogramming library [`Boost.Hana`][boost-hana-ref] to generate
  [`pybind11`][pybind11-bindings-with-boost-hana] bindings
* [MPI][wiki-mpi] and [`mpi4py`][mpi4py-ref] as building blocks for distributing computing
* [`pytest`][pytest-doc] for unit tests, e.g. [`edamer.fn.transpose`][py-edamer-fn-transpose-pytest] and
  [`edamer.fn.pca`][py-edamer-fn-pca-pytest]
* [CMake 3][cmake3-tut] and [`hbrs-cmake`][hbrs-cmake] to [build, export and install our library](CMakeLists.txt)
* [GitLab CI][hbrs-gitlab-py-edamer-ci] to continuously build and test our code

### Status Quo

So far, `Elemental`'s data structures for non-distributed and distributed matrices and its corresponding wrappers from
`hbrs-mpl` has been integrated with `NumPy`. For example, 2d NumPy arrays (matrices) can be converted into
non-distributed Elemental matrices and vice versa, without having to copy any matrix entry. Further, Elemental's MPI
interface has been integrated with `mpi4py`. This allows e.g. to define the MPI computation grid with mpi4py and then
hand it over Elemental. This conversion is done implicitly, i.e. a custom adapter takes care of converting mpi4py
communicators into MPI handles (as defined by the official MPI C API) that can be consumed by Elemental and vice versa.
Distributed Elemental matrices can be constructed from a MPI computation grid and local matrices. All of
[Elemental's matrix distributions][el-matrix-dists] are available in `edamer`, i.e. [`[MC,MR]`][el-dist-mc-mr],
[`[STAR,STAR]`][el-dist-star-star] and [`[VC,STAR]`][el-dist-vc-star].

The functionality for the Python and C++ interop is heavily based on [`pybind11`][pybind11-doc] and
[`Boost.Hana`][boost-hana-ref].

## How to build, install and run code using `Docker` or `Podman`

For a quick and easy start into developing with Python and C++, a set of ready-to-use `Docker`/`Podman` images
`jm1337/debian-dev-hbrs` and `jm1337/debian-dev-full` (supports more languages) has been created. They contain a full
development system including all tools and libraries necessary to hack on distributed decomposition algorithms and more
([Docker Hub][docker-hub-jm1337], [source files for Docker images][docker-artifacts]).

Sidenote:
> Creating the Docker images was tedious, especially because bugs ([`#959387`][debian-bug-959387],
> [`#972551`][debian-bug-972551]) in Debian's ParaView package (affected Ubuntu and derivates as well) had to be fixed
> or worked around, i.e. Debian's did not package the development files necessary to use e.g. ParaView Catalyst. But by
> now most of the proposed patches have been incorporated into Debian.

### Install `Docker` or `Podman`

* On `Debian 10 (Buster)` or `Debian 11 (Bullseye)` just run `sudo apt install docker.io`
  or follow the [official install guide][docker-install-debian] for Docker Engine on Debian
* On `Ubuntu 18.04 LTS (Bionic Beaver)` and `Ubuntu 20.04 LTS (Focal Fossa)` just run `sudo apt install docker.io`
  (from `bionic/universe` and `focal/universe` repositories)
  or follow the [official install guide][docker-install-ubuntu] for Docker Engine on Ubuntu
* On `Windows 10` follow the [official install guide][docker-install-windows] for Docker Desktop on Windows
* On `Mac` follow the [official install guide][docker-install-mac] for Docker Desktop on Mac
* On `Fedora`, `Red Hat Enterprise Linux (RHEL)` and `CentOS` follow the [official install guide][podman-install] for
  Podman

### Setup and run container

```sh
# docker version 18.06.0-ce or later is recommended
docker --version

# fetch docker image
docker pull jm1337/debian-dev-hbrs:bullseye

# log into docker container
docker run -ti jm1337/debian-dev-hbrs:bullseye
# or using a persistent home directory, e.g.
docker run -ti -v /HOST_DIR:/home/devil/ jm1337/debian-dev-hbrs:bullseye
# or using a persistent home directory on Windows hosts, e.g.
docker run -ti -v C:\YOUR_DIR:/home/devil/ jm1337/debian-dev-hbrs:bullseye
```

Podman strives for complete CLI compatibility with Docker, hence
[you may use the `alias` command to create a `docker` alias for Podman][docker-to-podman-transition]:
```sh
alias docker=podman
```

### Build and run code inside container

Execute the following commands within the `Docker`/`Podman` container:

```sh
# choose a compiler
export CC=clang-10
export CXX=clang++-10
# or
export CC=gcc-10
export CXX=g++-10

# fetch, compile and install prerequisites
git clone --depth 1 https://github.com/JM1/hbrs-cmake.git
cd hbrs-cmake
mkdir build && cd build/
# install to non-system directory because sudo is not allowed in this docker container
cmake \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local \
    ..
make -j$(nproc)
make install
cd ../../

git clone --depth 1 https://github.com/JM1/hbrs-mpl.git
cd hbrs-mpl
mkdir build && cd build/
cmake \
 -DCMAKE_INSTALL_PREFIX=$HOME/.local \
 -DHBRS_MPL_ENABLE_ELEMENTAL=ON \
 -DHBRS_MPL_ENABLE_MATLAB=OFF \
 -DHBRS_MPL_ENABLE_TESTS=OFF \
 -DHBRS_MPL_ENABLE_BENCHMARKS=OFF \
 ..
make -j$(nproc)
make install
cd ../../

# fetch, compile and install python-edamer
git clone --depth 1 https://github.com/JM1/python-edamer.git
cd python-edamer
mkdir build && cd build/
cmake \
    -DCMAKE_Python3_COMPILER_FORCED=ON \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local \
    -DEDAMER_ENABLE_SCALAR_INT=ON \
    -DEDAMER_ENABLE_SCALAR_FLOAT=ON \
    -DEDAMER_ENABLE_SCALAR_DOUBLE=ON \
    -DEDAMER_ENABLE_SCALAR_COMPLEX_FLOAT=ON \
    -DEDAMER_ENABLE_SCALAR_COMPLEX_DOUBLE=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_CIRC_CIRC=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_MC_MR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_MC_STAR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_MD_STAR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_MR_MC=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_MR_STAR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_MC=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_MD=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_MR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_STAR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_VC=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_STAR_VR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_VC_STAR=ON \
    -DEDAMER_ENABLE_MATRIX_DISTRIBUTION_VR_STAR=ON \
    -DEDAMER_ENABLE_TESTS=ON \
    -DMPIEXEC_MAX_NUMPROCS=2 \
    ..
make -j$(nproc)
# If unit test dt_el_dist_matrix fails in function test_copy_redist due to zeros
# in the upper matrix indices, then try to use a different MPI point-to-point
# management layer, e.g. ob1 instead of ucx.
#export OMPI_MCA_pml=ob1
ctest --verbose --output-on-failure
make install

export LD_LIBRARY_PATH=$HOME/.local/lib
export PYTHONPATH=$HOME/.local/lib/python3/dist-packages
python3 -c "from edamer import detail, dt, fn; print('All systems go')"
```

For more examples on how to build and test this code see [`.gitlab-ci.yml`](.gitlab-ci.yml).

## Knowledge Base

### Why does compilation take so much time and memory? The compiled library is several hundreds of megabytes large!!!

Calling C++ functions from Python requires to declare and compile bindings with `pybind11` for all C++ function
signatures that should be callable from Python into the wrapper library. For example, the function
[`multiply`][py-edamer-fn-multiply] must be bound for Elemental matrices with entry type
[`int`, `float`, `double`, ...][py-edamer-dt-el-matrix-bindings].
If `multiply` is not wrapped for Elemental's matrices with `float` entries, then it cannot be used from Python.
Currently, `multiply` is wrapped for `int`, `float`, `double`, `complex<float>` and `complex<double>`. For
Elemental's non-distributed matrices, this results into five function overloads for `multiply` in Python:

```
1. multiply(a: ElMatrix_StdInt32T, b: ElMatrix_StdInt32T) -> ElMatrix_StdInt32T
2. multiply(a: ElMatrix_Float, b: ElMatrix_Float) -> ElMatrix_Float
3. multiply(a: ElMatrix_Double, b: ElMatrix_Double) -> ElMatrix_Double
4. multiply(a: ElMatrix_ElComplex_Float, b: ElMatrix_ElComplex_Float) -> ElMatrix_ElComplex_Float
5. multiply(a: ElMatrix_ElComplex_Double, b: ElMatrix_ElComplex_Double) -> ElMatrix_ElComplex_Double
```

For Elemental's distributed matrices it gets messy. For example, it should be possible to multiply a matrix with
`[STAR,STAR]` distribution and a matrix with `[MC,MR]` distribution. Hence `multiply` must be provided for all
possible combinations of matrix distributions, i.e. the cartesian product of `[int, float, double, ...]`,
`[[STAR,STAR],[MC,MR],[MR,MC], ...]` and `[[STAR,STAR],[MC,MR],[MR,MC], ...]`. This results into another `845`(!)
function overloads for `multiply`!

For each of these Python function overloads, a C++ compiler generates a separate code path, because
[function templates in C++ get instantiated][cpp-ref-class-template]. Generic code in other languages such as Java, is
compiled differently, e.g. Java implements generics using type erasure and generates code just once for all generic
functions. Both has advantages and disadvantages, i.e. a C++ compiler can apply optimizations a Java compiler cannot.
But with many template instantiations, C++ libraries might get huge in size. For all entry types and all matrix
distributions, the object file of `multiply` growths to `50MiB` in debug mode. The compilation and linking processes
take up to `6GiB` of RAM.

To reduce compilation time, memory usage and code size, irrelevant matrix entry types aka scalar types and matrix
distributions can be disabled at compile time using
[CMake options `EDAMER_ENABLE_SCALAR_*` and `EDAMER_ENABLE_MATRIX_DISTRIBUTION_*`][py-edamer-cmake-options].
But beware that disabled matrix template instantiations cannot be used as function arguments and function return values!
For example, if `transpose` is applied to a matrix with `[MC,MR]` distribution, then `transpose` will return a matrix
with `[MR,MC]` distribution. The returned matrix can only be used if this matrix distribution has been compiled in.
Accessing a return value with a type that has not been compiled in results in an runtime error.

### Unit test `dt_el_dist_matrix` fails in function `test_copy_redist` due to zeros in the upper matrix indices!?

Try to use a different MPI point-to-point management layer, e.g. `ob1` instead of `ucx`.
For example, set and export variable `OMPI_MCA_pml` before executing the unit tests:

```sh
export OMPI_MCA_pml=ob1
ctest --verbose
```

Reason for this error is still unknown. Till now, it only occurs sporadically and inside container
`jm1337/debian-dev-hbrs:bullseye`.

## License

GNU General Public License v3.0 or later

See [LICENCE.md](LICENSE.md) to see the full text.

## Author

Jakob Meng
@jm1 ([GitHub.com][github-jm1], [Web][jm])

[//]: # (References)

[boost-hana]: https://github.com/boostorg/hana
[boost-hana-ref]: https://boostorg.github.io/hana/
[cmake3-tut]: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
[cpp-ref]: https://en.cppreference.com/w/cpp
[cpp-ref-class-template]: https://en.cppreference.com/w/cpp/language/class_template
[debian-bug-959387]: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=959387
[debian-bug-972551]: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=972551
[docker-artifacts]: https://github.com/JM1/docker-artifacts
[docker-hub-jm1337]: https://hub.docker.com/r/jm1337/
[docker-install-debian]: https://docs.docker.com/engine/install/debian/
[docker-install-mac]: https://docs.docker.com/docker-for-mac/install/
[docker-install-ubuntu]: https://docs.docker.com/engine/install/ubuntu/
[docker-install-windows]: https://docs.docker.com/docker-for-windows/install/
[docker-to-podman-transition]: https://developers.redhat.com/blog/2020/11/19/transitioning-from-docker-to-podman/
[elemental]: https://github.com/elemental/Elemental
[el-matrix-dists]: https://github.com/elemental/elemental-web/blob/master/source/doc-dev/core/dist_matrix/DM.rst
[el-dist-vc-star]: https://github.com/elemental/elemental-web/blob/master/source/doc-dev/core/dist_matrix/Element/VC_STAR.rst
[el-dist-mc-mr]: https://github.com/elemental/elemental-web/blob/master/source/doc-dev/core/dist_matrix/Element/MC_MR.rst
[el-dist-star-star]: https://github.com/elemental/elemental-web/blob/master/source/doc-dev/core/dist_matrix/Element/STAR_STAR.rst
[fhg-gitlab-edamer]: https://gitlab.scai.fraunhofer.de/ndv/research/excellerat/edamer
[fhg-gitlab-py-edamer]: https://gitlab.scai.fraunhofer.de/ndv/research/excellerat/edamer-python-library
[fhg-scai]: https://www.scai.fraunhofer.de/
[github-jm1]: https://github.com/jm1
[hbrs-gitlab-hbrs-mpl]: https://git.inf.h-brs.de/jmeng2m/hbrs-mpl/
[hbrs-gitlab-py-edamer]: https://git.inf.h-brs.de/jmeng2m/python-edamer
[hbrs-gitlab-py-edamer-ci]: https://git.inf.h-brs.de/jmeng2m/python-edamer/-/pipelines
[hbrs-cmake]: https://github.com/JM1/hbrs-cmake/
[hbrs-mpl]: https://github.com/JM1/hbrs-mpl/
[jm]: http://www.jakobmeng.de
[matlab-help]: https://de.mathworks.com/help/
[mpi4py-ref]: https://mpi4py.readthedocs.io/
[nek5000]: https://nek5000.mcs.anl.gov/
[podman-install]: https://podman.io/getting-started/installation
[pv-catalyst-guide]: https://www.paraview.org/files/catalyst/docs/ParaViewCatalystUsersGuide_v2.pdf "The ParaView Catalyst Users Guide v2"
[py-edamer]: https://github.com/JM1/python-edamer
[py-edamer-cmake-options]: https://github.com/JM1/python-edamer/blob/master/CMakeLists.txt#L17
[py-edamer-dt-el-matrix-bindings]: https://github.com/JM1/python-edamer/blob/master/src/edamer/dt/el_matrix/impl.cpp#L122
[py-edamer-fn-multiply]: https://github.com/JM1/python-edamer/tree/master/src/edamer/fn/multiply
[py-edamer-fn]: https://github.com/JM1/python-edamer/tree/master/src/edamer/fn
[py-edamer-fn-pca-pytest]: https://github.com/JM1/python-edamer/blob/master/src/edamer/fn/pca/test/elemental.py
[py-edamer-fn-transpose-pytest]: https://github.com/JM1/python-edamer/blob/master/src/edamer/fn/transpose/test/elemental.py
[pybind11-doc]: https://pybind11.readthedocs.io/
[pybind11-bindings-with-boost-hana]: https://github.com/jwbuurlage/pybind11_plus_hana
[pytest-doc]: https://docs.pytest.org/
[python3-ref]: https://docs.python.org/3/reference/
[wiki-cpp17]: https://en.wikipedia.org/wiki/C++17
[wiki-mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface
