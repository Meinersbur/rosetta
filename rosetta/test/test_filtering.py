#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import argparse

from context import rosetta
from rosetta.registry import Benchmark
from rosetta.filtering import *


class TestFiltering(unittest.TestCase):
    def setUp(self):
        # Initialize test data
        self.benchmarks = [
            Benchmark("idioms.assign", "target1", "exepath1", "buildtype1", "serial", "configname1"),
            Benchmark("idioms.assign", "target2", "exepath2", "buildtype2", "cuda", "configname2"),
            Benchmark("polybench.cholesky", "target3", "exepath3", "buildtype3", "cuda", "configname3"),
            Benchmark("polybench.gemm", "target3", "exepath3", "buildtype3", "openmp-parallel", "configname3")
        ]
        self.reset_args()

    def reset_args(self):
        self.args = argparse.Namespace(
            filter_include_program_substr=[],
            filter_include_program_exact=[],
            filter_include_program_regex=[],
            filter_exclude_program_substr=[],
            filter_exclude_program_exact=[],
            filter_exclude_program_regex=[],
            filter_include_ppm_substr=[],
            filter_include_ppm_exact=[],
            filter_include_ppm_regex=[],
            filter_exclude_ppm_substr=[],
            filter_exclude_ppm_exact=[],
            filter_exclude_ppm_regex=[],
        )

    def test_get_filtered_benchmarks(self):
        # test: no filter
        self.reset_args()
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 4, "Length should be 4")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-exact cuda
        self.reset_args()
        self.args.filter_include_ppm_exact = ['cuda']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")

        # test: --filter-exclude-ppm-exact cuda
        self.reset_args()
        self.args.filter_exclude_ppm_exact = ['cuda']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-program-substr assign
        self.reset_args()
        self.args.filter_include_program_substr = ['assign']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-include-ppm-exact cuda --filter-exclude-program-exact polybench.cholesky
        self.reset_args()
        self.args.filter_include_ppm_exact = ['cuda']
        self.args.filter_exclude_program_exact = ['polybench.cholesky']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-exclude-program-exact polybench.cholesky --filter-include-ppm-exact cuda
        self.reset_args()
        self.args.filter_exclude_program_exact = ['polybench.cholesky']
        self.args.filter_include_ppm_exact = ['cuda']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-include-ppm-regex ".*l.*"
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*l.*"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-regex ".*l.*" --filter-include-ppm-exact cuda
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*l.*"]
        self.args.filter_include_ppm_exact = ["cuda"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 4, "Length should be 4")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-regex ".*l.*" --filter-include-ppm-exact cuda --filter-exclude-ppm-exact serial
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*l.*"]
        self.args.filter_include_ppm_exact = ["cuda"]
        self.args.filter_exclude_ppm_exact = ["serial"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 3, "Length should be 3")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-regex ".*a.*"
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*a.*"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 4, "Length should be 4")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-exact cuda --filter-include-program-exact idioms.assign
        self.reset_args()
        self.args.filter_include_ppm_exact = ["cuda"]
        self.args.filter_include_program_exact = ["idioms.assign"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-include-ppm-exact cuda --filter-include-program-exact idioms.assign --filter-exclude-ppm-exact serial
        self.reset_args()
        self.args.filter_include_ppm_exact = ['cuda']
        self.args.filter_include_program_exact = ['idioms.assign']
        self.args.filter_exclude_ppm_exact = ['serial']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

if __name__ == '__main__':
    unittest.main()
