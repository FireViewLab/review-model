"""리뷰별 집계 행을 받아 신호를 판정하고 층별 후보 풀을 만든다. 작성자는 해시로만 저장한다."""
import csv, hashlib, heapq, hmac, os, re, sys
pf, out = sys.argv[1], sys.argv[2]
salt = open(os.path.expanduser('~/.config/review-v4/author_salt')).read().strip().encode()
csv.field_size_limit(10**8)
R = {
 'promo': re.compile(r'(무조건|꼭|강력|적극|다들|모두)\s*(사세요|구매하세요|구입하세요|추천합니다|추천해요|쟁이세요)|강추|인생템|쟁여|사야\s*(돼|해)|후회\s*안|안\s*사면\s*손해|품절\s*전에|지금\s*당장'),
 'unused_a': re.compile(r'아직\s*(안|못)\s*(써|사용|먹|발라|입어)|사용\s*전|개봉\s*전|미개봉|안\s*써\s*봤|써보진|써\s*보지는|배송만|받기만'),
 'unused_b': re.compile(r'효과|좋을\s*것|좋아요|최고|추천|만족'),
 'certain': re.compile(r'100\s*%|백\s*프로|누구나|모든\s*(분|사람)|효과\s*(보장|확실)|확실한\s*효과|무조건\s*효과'),
 'repeat': re.compile(r'([가-힣A-Za-z]{2,})(\s*\1){3,}'),
 'spec': re.compile(r'(성분|함유|효능|특허|추출물|mAh|소재|원단|용량|기능|인증)'),
 'disclosed': re.compile(r'체험단|협찬|제공\s*받|원고료|지원\s*받|서포터즈|무상\s*(으로)?\s*제공|광고\s*(입니다|포함)|업체(로부터|에서)\s*(제공|지원)|브랜드(로부터|에서)\s*(제공|지원)|소정의|리뷰\s*이벤트'),
}
STRATA = ['combo','promo','unused','certain','repeat','spec','disclosed','burst','network','author']
def flags(r):
    c = r['content']; mlen = int(r['mlen'] or 0); rating = float(r['rating']) if r['rating'] else None
    rep = R['repeat'].search(c)
    day_c = int(r['day_c'] or 0); mean = float(r['day_mean'] or 0)
    return {
     'combo': rating == 5 and int(r['imgs'] or 0) == 0 and mlen < 10,
     'promo': bool(R['promo'].search(c)),
     'unused': bool(R['unused_a'].search(c)) and bool(R['unused_b'].search(c)),
     'certain': bool(R['certain'].search(c)),
     'repeat': bool(rep) and len(set(rep.group(1))) >= 2,
     'spec': mlen >= 80 and len(R['spec'].findall(c)) >= 3,
     'disclosed': bool(R['disclosed'].search(c)),
     'burst': day_c >= 10 and day_c >= 5 * mean and (rating is None or rating == 5),
     'network': int(r['dup_ids'] or 0) >= 2,
     'author': int(r['author_n'] or 0) >= 30 and int(r['author_prods'] or 0) >= 20,
    }
def key(rid, tag): return hashlib.md5(f'{rid}{tag}'.encode()).hexdigest()
heaps = {s: [] for s in STRATA + ['random']}
counts = {s: 0 for s in STRATA}; total = 0; anys = 0
seen = set()
for r in csv.DictReader(sys.stdin):
    # 같은 리뷰가 여러 옵션 상품에 공유되어 저장된 경우 한 번만 센다.
    if r['review_id'] in seen:
        continue
    seen.add(r['review_id'])
    total += 1
    fl = flags(r)
    a = r.pop('author') or ''
    r['author_hash'] = '' if not a else ('masked-' if '*' in a else '') + hmac.new(salt, f'{pf}|{a}'.encode(), hashlib.sha256).hexdigest()[:12]
    for s in STRATA: r[f's_{s}'] = 't' if fl[s] else 'f'
    if any(fl[s] for s in STRATA if s != 'disclosed'): anys += 1
    for s in STRATA:
        if fl[s]:
            counts[s] += 1
            item = (-int(key(r['review_id'], 'v4-pool'), 16), total, dict(r, stratum=s))
            (heapq.heappush if len(heaps[s]) < 200 else heapq.heappushpop)(heaps[s], item)
    item = (-int(key(r['review_id'], 'v4-rand'), 16), total, dict(r, stratum='random'))
    (heapq.heappush if len(heaps['random']) < 600 else heapq.heappushpop)(heaps['random'], item)
with open(f'{out}/{pf}_counts.csv', 'w') as fh:
    fh.write('count_total,' + ','.join(STRATA) + ',any_suspicious\n')
    fh.write(f'{total},' + ','.join(str(counts[s]) for s in STRATA) + f',{anys}\n')
fields = ['platform','stratum','product_id','review_id','content','rating','author_hash','author_n','author_prods','written_date','option','imgs','helpful_count','mlen','day_c','day_mean','prod_total','dup_ids','dup_prods'] + [f's_{s}' for s in STRATA]
n = 0
with open(f'{out}/{pf}.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore'); w.writeheader()
    for s, h in heaps.items():
        for _, _, r in sorted(h, reverse=True): w.writerow(r); n += 1
print(pf, 'total', total, 'pool', n, flush=True)
