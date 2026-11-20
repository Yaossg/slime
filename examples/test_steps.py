#!/usr/bin/env python3
"""Compare complete tuples to an independent Python Java-arithmetic oracle."""
import argparse
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parent
GPU = False
A = 0x5DEECE66D
MASK = (1 << 48) - 1
XOR = 987234911


def i32(v):
    v &= (1 << 32) - 1
    return v - (1 << 32) if v >> 31 else v


def term(x, z):
    return i32(x*x*4987142) + i32(x*5947611) + i32(z*z)*4392871 + i32(z*389711)


def first(seed, x, z):
    state = ((seed + term(x, z)) ^ XOR ^ A) & MASK
    return (state*A + 11) & MASK


def slime(seed, x, z):
    state = first(seed, x, z)
    while True:
        q = state >> 17
        if q < 2147483640:
            return q % 10 == 0
        state = (state*A + 11) & MASK


def oracle(seed, seeds, x0, z0, width, height, n):
    return [[s, x, z] for s in range(seed, seed+seeds)
            for x in range(x0, x0+width) for z in range(z0, z0+height)
            if all(slime(s, x+dx, z+dz) for dx in range(n) for dz in range(n))]


def execute(binary, args, json_output=True):
    result = subprocess.run([str(ROOT/'bin'/binary), *map(str, args)],
                            capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout) if json_output else result.stdout


def cli(case):
    seed, seeds, x, z, width, height, n = case
    return ['--seed', seed, '--seeds', seeds, '--x0', x, '--z0', z,
            '--width', width, '--height', height, '-n', n, '--repeats', 1]


class Steps(unittest.TestCase):
    def compare(self, case):
        expected = oracle(*case)
        self.assertEqual(execute('cpu_forward', cli(case))['hits'], expected)
        if GPU:
            result = execute('cuda_steps', cli(case))
            self.assertTrue(result['comparison_equal'])
            self.assertEqual(result['hits'], expected)

    def test_nonzero_seed_and_nonblock_dimensions(self):
        self.compare((1337, 7, -23, 17, 19, 21, 1))

    def test_negative_and_overflow_coordinates(self):
        self.compare((98765, 3, -1875000, 1874995, 17, 19, 2))

    def test_int32_edges(self):
        self.compare((42, 2, -2147483648, 2147483600, 17, 17, 2))

    def test_known_hit_and_partial_tiles(self):
        case = (61692151016558, 3, 51050, 419107, 37, 35, 5)
        self.compare(case)
        self.assertIn([61692151016558, 51069, 419124], oracle(*case))

    def test_all_pattern_sizes(self):
        for n in range(1, 9):
            with self.subTest(n=n):
                self.compare((61692151016558, 2, 51069, 419124, 3, 3, n))

    def test_high16_equivalence_and_seed48_wrap(self):
        base = (61692151016558, 2, 51060, 419115, 17, 19, 5)
        high = (base[0]+(34567 << 48), *base[1:])
        self.compare(high)
        # Compare normalized seed labels, not raw world seed IDs.
        normalized = [[s & MASK, x, z] for s, x, z in oracle(*high)]
        self.assertEqual(normalized, oracle(*base))
        self.compare((MASK-2, 7, -7, 5, 17, 19, 1))

    def test_rejection_tail(self):
        # Engineer each of Java's eight rejected first outputs and a successful redraw.
        inv = pow(A, -1, 1 << 48)
        for q in range(2147483640, 2147483648):
            for suffix in range(10000):
                v = (q << 17) | suffix
                u = ((v-11)*inv) & MASK
                seed = ((u ^ (XOR ^ A))-term(-31, 29)) & MASK
                if slime(seed, -31, 29):
                    break
            else:
                self.fail('could not construct successful redraw')
            self.assertEqual(first(seed, -31, 29) >> 17, q)
            self.compare((seed, 1, -31, 29, 1, 1, 1))

    def test_invalid_ranges_and_truncation(self):
        invalid = [
            ['--width', 0], ['--repeats', 0], ['--seeds', 0],
            ['--seed', (1 << 64)-1, '--seeds', 2],
            ['--x0', 2147483647, '--width', 2],
            ['--seed', 0, '--seeds', 5, '-n', 1, '--capacity', 1],
        ]
        for binary in ['cpu_forward'] + (['cuda_steps'] if GPU else []):
            for args in invalid:
                p = subprocess.run([str(ROOT/'bin'/binary), *map(str, args)],
                                   capture_output=True, text=True, timeout=60)
                self.assertNotEqual(p.returncode, 0, (binary, args))

    def test_outer_tall_rectangle_fallback(self):
        if not GPU:
            self.skipTest('requires --gpu')
        # More than 65535 tiles on Z: exercise the safe linear-grid fallback.
        args = cli((61692151016558, 1, 51069, 0, 1, 1048561, 5))
        cpu = execute('cpu_forward', args)
        gpu = execute('cuda_steps', [*args, '--modes', '3,4'])
        self.assertEqual(cpu['hits'], gpu['hits'])
        self.assertIn([61692151016558, 51069, 419124], cpu['hits'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args, rest = parser.parse_known_args()
    GPU = args.gpu
    unittest.main(argv=[__file__, *rest], verbosity=2)
