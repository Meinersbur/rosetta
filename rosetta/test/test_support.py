#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
from context import rosetta
from rosetta.util.support import *


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Enter class tests")

    @classmethod
    def tearDownClass(cls):
        print("Exit class tests")

    def setUp(self):
        print("class test")

    def tearDown(self):
        print("Exit test")

    def test_removeprefix(self):
        self.assertEqual(removeprefix("prefixmy", "prefix"), "my")
        self.assertEqual(removeprefix("mynofix", "prefix"), "mynofix")

    def test_removesuffix(self):
        self.assertEqual(removesuffix("mysuffix", "suffix"), "my")
        self.assertEqual(removesuffix("mynofix", "suffix"), "mynofix")

    def test_cached_generator(self):
        class Testclass:
            def __init__(self):
                self.called = 0

            @cached_generator
            def cacheme(self):
                self.called += 1
                yield "one"
                yield "two"

        obj = Testclass()
        self.assertEqual(obj.called, 0)
        self.assertSequenceEqual(obj.cacheme, ["one", "two"])
        self.assertEqual(obj.called, 1)
        self.assertSequenceEqual(obj.cacheme, ["one", "two"])
        self.assertEqual(obj.called, 1)

        # Should work like cache_property if not a generator
        class Testclass2:
            def __init__(self):
                self.called = 0

            @cached_generator
            def cacheme(self):
                self.called += 1
                return ["eins", "zwei"]

        obj = Testclass2()
        self.assertEqual(obj.called, 0)
        self.assertSequenceEqual(obj.cacheme, ["eins", "zwei"])
        self.assertEqual(obj.called, 1)
        self.assertSequenceEqual(obj.cacheme, ["eins", "zwei"])
        self.assertEqual(obj.called, 1)


if __name__ == '__main__':
    unittest.main()
