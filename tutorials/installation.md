How to build Lego library
========

Lego library is designed to be compilable on Linux, MacOS and Window platforms. But currently, I only have tested it on Linux.

## Building on Linux platform

You need to have cmake >= 3.10 and a C++ compiler that supports C++17

First clone this github repo:
```shell script
git clone https://github.com/cirahnitex/lego.git
```

Create a temporary build directory:
```shell script
mkdir -pv lego/build
cd lego/build
```

perform compilation in the temporary build directory:
```shell script
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Run the test program to verify that the library has been built properly:
```shell script
./xor_demo
```

Install the library to your machine.
```shell script
make install
```
After this step, all the required header files will be installed (default installation path: `/usr/local/include/`) and two required shared libraries, `liblego.so` and `libdynet.so` will be released (default installation path: `/usr/local/lib/`)

### CUDA support

If you have CUDA installed, you can build with CUDA by adding a cmake flag like this:
```shell script
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKEND=cuda
```

You can also specify the path to CUDA toolkit if you have multiple versions of CUDA installed or your CUDA is not installed in the default location.
```shell script
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKEND=cuda -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
```
