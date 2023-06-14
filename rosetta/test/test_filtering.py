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
            Benchmark("apple", "target1", "exepath1", "buildtype1", "green", "configname1"),
            Benchmark("apple", "target2", "exepath2", "buildtype2", "red", "configname2"),
            Benchmark("cherry", "target3", "exepath3", "buildtype3", "red", "configname3"),
            Benchmark("orange", "target3", "exepath3", "buildtype3", "orange", "configname3")
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

        # test: --filter-include-ppm-exact red
        self.reset_args()
        self.args.filter_include_ppm_exact = ['red']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")

        # test: --filter-exclude-ppm-exact red
        self.reset_args()
        self.args.filter_exclude_ppm_exact = ['red']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-exact red --filter-exclude-program-exact cherry
        self.reset_args()
        self.args.filter_include_ppm_exact = ['red']
        self.args.filter_exclude_program_exact = ['cherry']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-exclude-program-exact cherry --filter-include-ppm-exact red
        self.reset_args()
        self.args.filter_exclude_program_exact = ['cherry']
        self.args.filter_include_ppm_exact = ['red']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-include-ppm-regex ".*n.*"
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*n.*"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 2, "Length should be 2")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-regex ".*n.*" --filter-include-ppm-exact red
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*n.*"]
        self.args.filter_include_ppm_exact = ["red"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 4, "Length should be 4")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-regex ".*n.*" --filter-include-ppm-exact red --filter-exclude-ppm-exact green
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*n.*"]
        self.args.filter_include_ppm_exact = ["red"]
        self.args.filter_exclude_ppm_exact = ["green"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 3, "Length should be 3")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-regex ".*e.*"
        self.reset_args()
        self.args.filter_include_ppm_regex = [".*e.*"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 4, "Length should be 4")
        self.assertTrue(self.benchmarks[0] in filtered_benchmarks, "Should match first benchmark")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")
        self.assertTrue(self.benchmarks[2] in filtered_benchmarks, "Should match third benchmark")
        self.assertTrue(self.benchmarks[3] in filtered_benchmarks, "Should match fourth benchmark")

        # test: --filter-include-ppm-exact red --filter-include-program-exact apple
        self.reset_args()
        self.args.filter_include_ppm_exact = ["red"]
        self.args.filter_include_program_exact = ["apple"]
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

        # test: --filter-include-ppm-exact red --filter-include-program-exact apple --filter-exclude-ppm-exact green
        self.reset_args()
        self.args.filter_include_ppm_exact = ['red']
        self.args.filter_include_program_exact = ['apple']
        self.args.filter_exclude_ppm_exact = ['green']
        filtered_benchmarks = get_filtered_benchmarks(self.benchmarks, self.args)
        self.assertEqual(len(filtered_benchmarks), 1, "Length should be 1")
        self.assertTrue(self.benchmarks[1] in filtered_benchmarks, "Should match second benchmark")

if __name__ == '__main__':
    unittest.main()
