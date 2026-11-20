#!/usr/bin/env python3
"""Check complete reverse variants and coordinate reuse with an independent oracle."""
import json
import re
from pathlib import Path
import subprocess
import statistics
import tempfile

from test_steps import slime

HERE = Path(__file__).resolve().parent


def run(binary, *args):
    result = subprocess.run(
        [str(HERE / 'bin' / binary), *map(str, args)],
        capture_output=True, text=True, check=True, timeout=180,
    )
    return result.stdout


def main():
    reverse = json.loads(run('reverse', 2))
    reference = json.loads((HERE / 'data' / 'known-5x5.json').read_text())
    assert reverse['anchor'] == reference['anchor']
    assert reverse['roots'] == reference['roots'] and reverse['comparison_equal']
    assert len(reverse['rows']) == 5
    assert all(len(row['samples_ms']) == 2 for row in reverse['rows'])
    assert all(abs(row['median_ms'] - statistics.median(row['samples_ms'])) < 1e-5
               for row in reverse['rows'])
    seeds = reverse['matches']
    assert len(seeds) == len(set(seeds)) == 24
    assert set(seeds) == set(reference['seeds'])
    assert 61692151016558 in seeds
    assert all(slime(seed, 51069 + x, 419124 + z)
               for seed in seeds for x in range(5) for z in range(5))
    with tempfile.TemporaryDirectory() as directory:
        positions = Path(directory) / 'positions.txt'
        # Repeated classes, negative coordinates and the known nonempty class.
        positions.write_text('51069 419124\n116605 550196\n-14467 288052\n0 0\n65536 131072\n')
        rows = [json.loads(line) for line in run('reuse', positions, 1).splitlines()]
        for invalid in ['51069 419124\n1\n', '51069 419124 extra\n',
                        '2147483647 0\n', '-2147483649 0\n']:
            positions.write_text(invalid)
            result = subprocess.run([str(HERE / 'bin' / 'reuse'), str(positions), '1'],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and ('input line' in result.stderr or 'fit int32' in result.stderr)
    timings = [row for row in rows if row['type'] == 'pipeline']
    assert {row['mode'] for row in timings} == {1, 2, 3}
    validation = next(row for row in rows if row['type'] == 'validation')
    assert validation['complete_roots_equal'] and validation['complete_matches_equal']
    hits = validation['matches']
    assert len(hits) == len(set(map(tuple, hits)))
    assert {seed for seed, x, z in hits if (x, z) == (51069, 419124)} == set(seeds)
    assert all(slime(seed, x + dx, z + dz)
               for seed, x, z in hits for dx in range(5) for dz in range(5))
    baseline = run('brute', '--x', 51069, '--z', 419124, '-n', 5,
                   '--start', 61692151016558, '--count', 16)
    count = int(re.search(r'\bmatches=(\d+)', baseline)[1])
    expected = sum(all(slime(seed, 51069 + dx, 419124 + dz)
                       for dx in range(5) for dz in range(5))
                   for seed in range(61692151016558, 61692151016558 + 16))
    assert count == expected
    for axis in ['--x', '--z']:
        result = subprocess.run([str(HERE / 'bin' / 'brute'), axis, '2147483647', '--count', '1'],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 2
    print(json.dumps({'reverse_seeds': len(seeds), 'reuse_matches': len(hits), 'passed': True}))


if __name__ == '__main__':
    main()
