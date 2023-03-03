#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from .util.support import *



class BuildConfig:
    def __init__(self, name, ppm, cmake_arg, cmake_def, compiler_arg, compiler_def, is_predefined=False,usecur=False):
        self.name = name
        self.ppm = set(ppm)
        self.cmake_arg = cmake_arg
        self.cmake_def = cmake_def
        self.compiler_arg = compiler_arg
        self.compiler_def = compiler_def
        self.is_predefined = is_predefined
        
        # Use the previous configuration instead of defining a new one
        self.usecur = usecur

        # TODO: select compiler executable

    def gen_cmake_args(self):
        compiler_args = self.compiler_arg.copy()
        for k, v in self.compiler_def:
            if v:
                compiler_args.append(f"-D{k}")
            else:
                compiler_args.append(f"-D{k}={v}")

        # TODO: Combine with (-D, -DCMAKE_<lang>_FLAGS) from compiler/cmake_arg
        cmake_opts = self.cmake_arg[:]
        for k, d in self.cmake_def.items():
            cmake_opts .append(f"-D{k}={d}")
        if compiler_args:
            # TODO: Only set the ones relevant for enable PPMs
            opt_args = shjoin(compiler_args)
            cmake_opts += [f"-DCMAKE_C_FLAGS={opt_args}", f"-DCMAKE_CXX_FLAGS={opt_args}",
                           f"-DCMAKE_CUDA_FLAGS={opt_args}"]  # TODO: Release flags?

        if self.ppm:
            # TODO: Case, shortcuts
            for ppm in ['serial', 'cuda', 'openmp-thread', 'openmp-task', 'openmp-target']:
                ucase_name = ppm.upper().replace('-', '_')
                if ppm in self.ppm:
                    cmake_opts.append(f"-DROSETTA_PPM_{ucase_name}=ON")
                else:
                    # TODO: switch to have default OFF, so we don't need to list all of them
                    cmake_opts.append(f"-DROSETTA_PPM_{ucase_name}=OFF")

        if self.name:
            cmake_opts.append(f"-DROSETTA_CONFIGNAME={self.name}")

        return cmake_opts


def make_buildconfig(name, ppm, cmake_arg, cmake_def, compiler_arg, compiler_def):
    return BuildConfig(name, ppm, cmake_arg, cmake_def, compiler_arg, compiler_def)

    