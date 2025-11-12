Sionna RT GUI
=============

An interactive GUI to create and visualize Sionna RT scenes, paths and radio maps.


Getting started
---------------

Tested on Ubuntu 24.04:

```bash
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -r ./requirements.txt
```

Then, start the viewer with:

```bash
python ./scripts/run.py
```


### Temporary workaround to display the radio map colorbar

Due to some missing Python bindings in Polyscope v2.5.0, if you would like the colorbar corresponding to the radio map to be displayed, you need to clone and install a slightly tweaked Polyscope version:

```bash
cd sionna-rt-gui
source ./.venv/bin/activate

cd ..
git clone --recursive ssh://git@gitlab-master.nvidia.com:12051/ubernerf-team/polyscope-py.git --branch imgui-images
cd polyscope-py
pip install -e .
```

### Building on the DGX Spark

Since Mitsuba does not ship wheels for non-Apple ARM CPUs, we have to make our own local build.

First, clone Mitsuba:

```bash
git clone --recursive https://github.com/mitsuba-renderer/mitsuba3.git
```

Then, make the following change in `ext/drjit/ext/drjit-core/ext/nanothread/ext/cmake-defaults/CMakeLists.txt`:

```
 if (SKBUILD)
    # Reasonably portable binaries for PyPI
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      # ARM64 architecture - use native or a baseline ARM arch
      set(${P}_NATIVE_FLAGS_DEFAULT "-march=armv8-a")
    else()
      # x86-64 architecture - use Ivy Bridge baseline
      set(${P}_NATIVE_FLAGS_DEFAULT "-march=ivybridge")
    endif() (edited)
```

Finally, build and install the relevant packages into your virtual environment:

```bash
cd sionna-rt-gui
source ./.venv/bin/activate

cd ../mitsuba3/ext/drjit
pip install -e .
cd ../nanobind
pip install -e .
pip install scikit-build-core hatch-fancy-pypi-readme
cd ../
# This step may take a while
CMAKE_ARGS="-DMI_DEFAULT_VARIANTS=\"scalar_rgb,scalar_spectral,scalar_spectral_polarized,llvm_ad_rgb,llvm_ad_mono,llvm_ad_mono_polarized,llvm_ad_spectral,llvm_ad_spectral_polarized,cuda_ad_rgb,cuda_ad_mono,cuda_ad_mono_polarized,cuda_ad_spectral,cuda_ad_spectral_polarized\"" pip install -e . --no-build-isolation
```

Then, you should be able to install the GUI's requirements and run it as normal:

```bash
cd sionna-rt-gui
pip install -r ./requirements.txt

python ./scripts/run.py
```
