#!/usr/bin/env bash
#
# libGaLG

CMAKE_VERSION=""
API_VERSION=""
MINOR_VERSION=""

DEV_INSTALL=false
NVCC_INSTALL=false
CMAKE_INSTALL=false

SKIP_TEST=false

function echoerr {
  red="\033[00;31m"
  restore='\033[0m'
  echo -e "${red}GaLG:Error: ${restore}$@${restore}" 1>&2
  exit 1
}

function echowarn {
  yellow="\033[00;33m"
  restore='\033[0m'
  echo -e "${yellow}GaLG:Warn: ${restore}$@${restore}"
}

function echoinfo {
  green="\033[00;32m"
  restore='\033[0m'
  echo -e "${green}GaLG:Info: ${restore}$@${restore}"
}

function echolog {
  restore='\033[0m'
  echo -e "${restore}${restore}$@${restore}"
}

function check_version {
  regex="[0-9]+(.[0-9]+)?"
  [[ $(grep -i 'CMAKE_MINIMUM_REQUIRED' CMakeLists.txt) =~ $regex ]] \
    && CMAKE_VERSION=$BASH_REMATCH
  regex="[0-9]+"
  [[ $(grep -i 'PROJECT_API_VERSION' CMakeLists.txt) =~ $regex ]] \
    && API_VERSION=$BASH_REMATCH
  regex="[0-9]+.[0-9]+"
  [[ $(grep -i 'PROJECT_MINOR_VERSION' CMakeLists.txt) =~ $regex ]] \
    && MINOR_VERSION=$BASH_REMATCH

  if [ -z "$CMAKE_VERSION" ] || [ -z "$API_VERSION" ] || [ -z "$MINOR_VERSION" ]; then
    echoerr "project's version can not be recognized"
  fi

  echolog "cmake required: version $CMAKE_VERSION"
  echolog "api version:    $API_VERSION"
  echolog "minor version:  $MINOR_VERSION"
}

function check_build_tools {
  if ! gcc_loc="$(type -p gcc)" || [ -z "$gcc_loc" ]; then
    echowarn "gcc is not installed"
    DEV_INSTALL=true
  else
    echolog "gcc:    $(type -p gcc)"
  fi

  if ! gpp_loc="$(type -p g++)" || [ -z "$gpp_loc" ]; then
    echowarn "g++ is not installed"
    DEV_INSTALL=true
  else
    echolog "g++:    $(type -p g++)"
  fi

  if ! nvcc_loc="$(type -p g++)" || [ -z "$nvcc_loc" ]; then
    echowarn "nvcc is not installed"
    NVCC_INSTALL=true
  else
    echolog "nvcc:   $(type -p nvcc)"
  fi

  if ! make_loc="$(type -p make)" || [ -z "$make_loc" ]; then
    echowarn "make is not installed"
    DEV_INSTALL=true
  else
    echolog "make:   $(type -p make)"
  fi

  if ! cmake_loc="$(type -p cmake)" || [ -z "$cmake_loc" ]; then
    echowarn "cmake is not installed"
    CMAKE_INSTALL=true
  else
    echolog "cmake:  $(type -p cmake)"
  fi

  if [ $DEV_INSTALL = true -o $NVCC_INSTALL = true -o $CMAKE_INSTALL = true ]; then
    echoerr "build tools are missing"
  fi
}

function bootstrap {
  echoinfo "bootstrapping libGaLG build system"
  check_version

  echoinfo "check build tools"
  check_build_tools

  echoinfo "generating makefiles, out of source building"
  mkdir -p build && cd build
  cmake ..
  echoinfo "build"
  make

  if [ $SKIP_TEST = false ]; then
    echoinfo "test"
    make test
  fi

  echoinfo "install"
  make install
}

bootstrap