#!/usr/bin/env python3
import argparse,json,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
BIN=HERE/'bin/slime-bench'
def run(*args,ok=True):
    p=subprocess.run([str(BIN),*map(str,args)],capture_output=True,text=True,timeout=180)
    if ok:
        assert p.returncode==0,p.stderr
        return json.loads(p.stdout)
    assert p.returncode!=0,p.stdout

def term(x,z):
    def i32(v):return ((v+2**31)%2**32)-2**31
    return i32(x*x*4987142)+i32(x*5947611)+i32(z*z)*4392871+i32(z*389711)
def slime(seed,x,z):
    s=((seed+term(x,z))^987234911^0x5DEECE66D)&((1<<48)-1)
    while True:
        s=(s*0x5DEECE66D+11)&((1<<48)-1);q=s>>17
        if q<2147483640:return q%10==0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',action='store_true');args=ap.parse_args()
    for radius in [0,1,2,31,1000000]:
        r=run('--x',-123,'--z',456,'--radius',radius,'--size',6,'--dry-run')
        assert r['anchors']==1+2*radius*(radius+1)
        assert r['first_anchor']==[-123,456-radius] and r['last_anchor']==[-123,456+radius]
    for bad in [[],['--radius','-1'],['--radius','1','--size','0'],['--radius','1','--size','7'],['--radius','abc'],['--radius','0','--x','2147483647'],['--radius','1','--repeat','0'],['--radius','1','--position-batch','0']]:run(*bad,'--dry-run',ok=False)
    points=[(dx,dz) for dz in range(-3,4) for dx in range(-3,4) if abs(dx)+abs(dz)<=3]
    for start in range(len(points)):
        count=min(4,len(points)-start)
        r=run('--radius',3,'--start-anchor',start,'--anchor-count',count,'--dry-run')
        assert r['start_anchor']==start and r['anchor_count']==count and r['anchors']==25
        assert r['first_anchor']==list(points[start]) and r['last_anchor']==list(points[start+count-1])
    assert run('--radius',3,'--start-anchor',24,'--dry-run')['anchor_count']==1
    for opts in [('--start-anchor',25),('--anchor-count',0),('--anchor-count',26),
                 ('--start-anchor',24,'--anchor-count',2),('--start-anchor',18446744073709551615)]:
        run('--radius',3,*opts,'--dry-run',ok=False)
    pattern='?##./#..?/..#?'
    r=run('--radius',0,'--pattern',pattern,'--dry-run')
    assert r['pattern']==pattern.split('/') and r['square_size'] is None
    assert (r['pattern_width'],r['pattern_height'],r['slime_cells'],r['non_slime_cells'])==(4,3,4,5)
    for bad_pattern in ['', '.', '??', '#/##', '#x', '#//.', '#/'+'#'*65]:
        run('--radius',0,'--pattern',bad_pattern,'--dry-run',ok=False)
    run('--radius',0,'--size',5,'--pattern','#','--dry-run',ok=False)
    run('--radius',0,'--pattern','#','--pattern','#','--dry-run',ok=False)
    run('--radius',0,'--pattern-file','/nonexistent-pattern','--dry-run',ok=False)
    run('--radius',0,'--x',2147483647,'--pattern','?#','--dry-run',ok=False)
    r=run('--radius',0,'--x',2147483647,'--pattern','#/#','--dry-run')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'pattern.txt';path.write_bytes(b'?##.\r\n#..?\r\n..#?\r\n')
        assert run('--radius',0,'--pattern-file',path,'--dry-run')['pattern']==pattern.split('/')
    heart_fixture=json.loads((HERE/'testdata/heart-5x5.json').read_text())
    assert (HERE/'patterns/heart-5x5.txt').read_text().splitlines()==heart_fixture['pattern']
    assert all(slime(heart_fixture['seed48'],heart_fixture['chunk_x']+dx,heart_fixture['chunk_z']+dz)==(ch=='#')
               for dz,row in enumerate(heart_fixture['pattern']) for dx,ch in enumerate(row))
    summary={'cli_scope_and_errors':True,'pattern_inputs':True}
    if args.gpu:
        heart_poc=run('--x',heart_fixture['chunk_x'],'--z',heart_fixture['chunk_z'],'--radius',0,
                      '--pattern-file',HERE/'patterns/heart-5x5.txt','--max-results',2000)
        heart_hits=heart_poc['rounds'][0]
        assert heart_hits['matches']==heart_fixture['matches']
        assert heart_hits['root_jobs']==heart_fixture['root_jobs']
        assert not heart_hits['sample_truncated'] and len(heart_hits['sample'])==heart_hits['matches']
        assert len(set(map(tuple,heart_hits['sample'])))==heart_hits['matches']
        assert [heart_fixture['seed48'],heart_fixture['chunk_x'],heart_fixture['chunk_z']] in heart_hits['sample']
        for seed,x,z in heart_hits['sample']:
            assert all(slime(seed,x+dx,z+dz)==(ch=='#') for dz,row in enumerate(heart_fixture['pattern']) for dx,ch in enumerate(row))
        summary['heart_pattern_verified_matches']=heart_hits['matches']
        # A mixed-mask regression fixture; this is not the target creeper pattern.
        fixture=json.loads((HERE/'testdata/mixed-mask-regression.json').read_text())
        face=run('--x',fixture['chunk_x'],'--z',fixture['chunk_z'],'--radius',0,
                 '--pattern','/'.join(fixture['pattern']),'--max-results',1000)
        hits=face['rounds'][0]
        assert hits['matches']==fixture['matches']
        assert not hits['sample_truncated'] and any(s==fixture['seed48'] for s,x,z in hits['sample'])
        for seed,x,z in hits['sample']:
            assert all(slime(seed,x+dx,z+dz)==(ch=='#') for dz,row in enumerate(fixture['pattern']) for dx,ch in enumerate(row))
        summary['mixed_mask_verified_matches']=hits['matches']
        common=['--x',51069,'--z',419124]
        reference=run(*common,'--radius',0,'--size',5,'--repeat',2,'--max-results',100)
        expected={61692151016558,15959335054446}
        assert all(r['matches']==24 and r['root_jobs']==84 for r in reference['rounds'])
        samples=[set(map(tuple,r['sample'])) for r in reference['rounds']]
        assert samples[0]==samples[1] and expected <= {s for s,x,z in samples[0]}
        sliced=run(*common,'--radius',0,'--size',5,'--job-batch',17,'--max-results',100,'--cache-classes',0)
        assert set(map(tuple,sliced['rounds'][0]['sample']))==samples[0]
        partial=run(*common,'--radius',0,'--size',5,'--max-results',3)
        assert partial['rounds'][0]['matches']==24 and len(partial['rounds'][0]['sample'])==3 and partial['rounds'][0]['sample_truncated']
        for n in [4,6]:
            r=run(*common,'--radius',0,'--size',n,'--max-results',1000)
            for seed,x,z in r['rounds'][0]['sample']:assert all(slime(seed,x+dx,z+dz) for dx in range(n) for dz in range(n))
            if n>=5:
                wanted={t for t in samples[0] if all(slime(t[0],t[1]+dx,t[2]+dz) for dx in range(n) for dz in range(n))}
                assert set(map(tuple,r['rounds'][0]['sample']))==wanted
            summary[f'size_{n}']=r['rounds'][0]['matches']
        # The same 5x5 positive constraints, shifted inside a non-square mask.
        # Empty anchor exercises normalization independently of the first #.
        rows=['?'*7]+['?'+'#'*5+'.' for _ in range(5)]+['?.?????']
        pattern='/'.join(rows)
        masked=run('--x',51068,'--z',419123,'--radius',0,'--pattern',pattern,'--max-results',100)
        wanted={(seed,51068,419123) for seed,x,z in samples[0]
                if all(ch=='?' or slime(seed,51068+dx,419123+dz)==(ch=='#')
                       for dz,row in enumerate(rows) for dx,ch in enumerate(row))}
        assert wanted and set(map(tuple,masked['rounds'][0]['sample']))==wanted
        assert masked['rounds'][0]['matches']==len(wanted)
        # Identical positive subset, but ignored rather than forbidden background.
        ignored=pattern.replace('.', '?')
        loose=run('--x',51068,'--z',419123,'--radius',0,'--pattern',ignored,'--max-results',100)
        assert {s for s,x,z in loose['rounds'][0]['sample']}=={s for s,x,z in samples[0]}
        # Force a forbidden cell to reject at least one known positive candidate.
        seed=next(iter(samples[0]))[0]
        dx,dz=next((dx,dz) for dz in range(16) for dx in range(16)
                   if not (1<=dx<=5 and 1<=dz<=5) and slime(seed,51068+dx,419123+dz))
        rejection=[list('?'*16) for _ in range(16)]
        for z in range(1,6):
            for x in range(1,6):rejection[z][x]='#'
        rejection[dz][dx]='.'
        rejected=run('--x',51068,'--z',419123,'--radius',0,'--pattern',
                     '/'.join(''.join(row) for row in rejection),'--max-results',100)
        expected_seeds={s for s,x,z in samples[0] if not slime(s,51068+dx,419123+dz)}
        assert len(expected_seeds)<24
        assert {s for s,x,z in rejected['rounds'][0]['sample']}==expected_seeds
        assert rejected['rounds'][0]['matches']==len(expected_seeds)
        summary['mixed_pattern_vs_independent_filter']=len(wanted)
        summary['forbidden_cell_excludes_candidates']=True
        diamond=run(*common,'--radius',2,'--size',5,'--max-results',1000,'--position-batch',3)
        split=[]
        for dz in range(-2,3):
            for dx in range(-2,3):
                if abs(dx)+abs(dz)>2:continue
                r=run('--x',51069+dx,'--z',419124+dz,'--radius',0,'--size',5,'--max-results',1000)
                split.extend(map(tuple,r['rounds'][0]['sample']))
        assert sorted(split)==sorted(map(tuple,diamond['rounds'][0]['sample']))
        assert diamond['anchors']==13
        partition=[]
        for start,count in [(0,4),(4,5),(9,4)]:
            r=run(*common,'--radius',2,'--start-anchor',start,'--anchor-count',count,
                  '--max-results',1000,'--position-batch',3)
            assert sum(g['anchors'] for g in r['rounds'][0]['gpus'])==count
            partition.extend(map(tuple,r['rounds'][0]['sample']))
        assert sorted(partition)==sorted(split)

        run('--radius',0,'--devices','0,0',ok=False)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'report.json'
            r=run(*common,'--radius',0,'--output',out)
            assert json.loads(out.read_text())==r
            run(*common,'--radius',0,'--output',out,ok=False)
        summary['gpu_known_matches']=24;summary['diamond_vs_individual_anchors']=True
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
