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
        self.assertEqual( removeprefix("prefixmy", "prefix") , "my")
        self.assertEqual(removeprefix("mynofix", "prefix"), "mynofix")

    def test_removesuffix(self):
        self.assertEqual( removesuffix("mysuffix", "suffix") , "my")
        self.assertEqual(removesuffix("mynofix", "suffix"), "mynofix")






if __name__ == '__main__':
    unittest.main()
