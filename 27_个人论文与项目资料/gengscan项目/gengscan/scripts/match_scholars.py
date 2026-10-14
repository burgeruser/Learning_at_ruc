# -*- coding: utf-8 -*-
"""Match Chinese scholars to OpenAlex authors and their Nature-family papers.

End-to-end pipeline:
  1. Build the Nature-family source set (journal -> source-id) once, cached to JSON.
  2. Three-way anchoring per scholar:
       pinyin name  +  institution (CN-preferred)  +  research field
     -> find the OpenAlex author, with a confidence label (high/medium/ambiguous).
  3. ORCID identity check: pull the matched author's ORCID and verify the name is
     consistent with the scholar's pinyin (catches wrong-author merges).
  4. Per-paper institution gate: for MEDIUM matches, keep only Nature papers where
     the author was actually affiliated with the scholar's institution on THAT paper.
     HIGH matches trust all their papers.

The four steps were originally separate scripts; merged here. Core logic
(anchoring thresholds, name-match rules, gating) is unchanged.

Network: uses curl with --noproxy '*' (OpenAlex/ORCID are reachable directly).
Dependency: pypinyin (pip install pypinyin).

Usage:
    # build the Nature source set (run once)
    python match_scholars.py build-sources --sources nature_sources.json

    # match scholars from a CSV/TSV (columns: name,org[,dept]) and write matched.json
    python match_scholars.py match --input scholars.csv --sources nature_sources.json --out matched.json

    # ORCID-verify the matched authors (updates matched.json in place)
    python match_scholars.py orcid --matched matched.json

    # per-paper institution gate for medium matches (updates matched.json in place)
    python match_scholars.py gate --matched matched.json --sources nature_sources.json
"""
import subprocess, json, urllib.parse, time, sys, os, argparse, csv
from pypinyin import lazy_pinyin

API = 'https://api.openalex.org'
UA = 'gengscan/1.0'


# ----------------------------------------------------------------------------
# curl helpers (bypass any system proxy; OpenAlex/ORCID are reachable directly)
# ----------------------------------------------------------------------------
def fetch(url):
    for _ in range(3):
        r = subprocess.run(['curl', '-s', '--noproxy', '*', '-m', '40',
                            '-H', 'User-Agent: ' + UA, url],
                           capture_output=True, text=True, encoding='utf-8')
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(1)
    return {'results': [], 'meta': {}}


def curl_raw(url, accept=None):
    cmd = ['curl', '-s', '--noproxy', '*', '-m', '30', '-H', 'User-Agent: ' + UA]
    if accept:
        cmd += ['-H', 'Accept: ' + accept]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8').stdout


def jget(url, accept=None):
    for _ in range(3):
        try:
            return json.loads(curl_raw(url, accept))
        except Exception:
            time.sleep(0.6)
    return None


def q(s):
    return urllib.parse.quote(s)


# ----------------------------------------------------------------------------
# Step 1: Nature-family source set
# ----------------------------------------------------------------------------
def build_nature_sources(path):
    ids = {}
    cur = '*'
    while cur:
        u = f'{API}/sources?search=Nature&per_page=200&select=id,display_name,issn_l&cursor={q(cur)}'
        r = fetch(u)
        for s in r['results']:
            dn = s['display_name'] or ''
            if dn == 'Nature' or dn.startswith('Nature ') or dn.startswith('Nature-'):
                ids[s['id'].split('/')[-1]] = dn
        cur = r['meta'].get('next_cursor')
        time.sleep(0.2)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    json.dump(ids, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return ids


def load_sources(path):
    nat = json.load(open(path, encoding='utf-8'))
    return nat, set(nat.keys())


# ----------------------------------------------------------------------------
# Step 2: three-way anchoring (pinyin + institution + field)
# ----------------------------------------------------------------------------
_inst_cache = {}


def inst_id(org_cn):
    if org_cn in _inst_cache:
        return _inst_cache[org_cn]
    r = fetch(f'{API}/institutions?search={q(org_cn)}&per_page=5&select=id,display_name,country_code')
    iid = None
    for x in r.get('results', []):
        if x.get('country_code') == 'CN':
            iid = (x['id'].split('/')[-1], x['display_name']); break
    if not iid and r.get('results'):
        x = r['results'][0]; iid = (x['id'].split('/')[-1], x['display_name'])
    _inst_cache[org_cn] = iid
    return iid


def pinyin_forms(name_cn):
    py = lazy_pinyin(name_cn)
    if len(py) < 2:
        return None, None, py
    surname = py[0].capitalize()
    given = ''.join(py[1:]).capitalize()
    return given, surname, py


# 中文院系/单位关键词 → OpenAlex 学科领域映射(领域锚定用)
DEPT_FIELDS = [
    ('材料', {'Materials Science', 'Engineering', 'Chemistry'}),
    ('化工', {'Chemical Engineering', 'Chemistry', 'Engineering'}),
    ('化学', {'Chemistry', 'Chemical Engineering', 'Materials Science'}),
    ('物理', {'Physics and Astronomy', 'Materials Science'}),
    ('天文', {'Physics and Astronomy', 'Earth and Planetary Sciences'}),
    ('数学', {'Mathematics'}),
    ('统计', {'Mathematics', 'Decision Sciences', 'Computer Science'}),
    ('力学', {'Engineering', 'Physics and Astronomy', 'Materials Science'}),
    ('计算机', {'Computer Science', 'Engineering'}),
    ('软件', {'Computer Science', 'Engineering'}),
    ('人工智能', {'Computer Science', 'Engineering'}),
    ('数据', {'Computer Science', 'Mathematics', 'Decision Sciences'}),
    ('网络空间', {'Computer Science', 'Engineering'}),
    ('集成电路', {'Engineering', 'Physics and Astronomy', 'Computer Science', 'Materials Science'}),
    ('微电子', {'Engineering', 'Physics and Astronomy', 'Materials Science'}),
    ('电子', {'Engineering', 'Computer Science', 'Physics and Astronomy'}),
    ('光电', {'Physics and Astronomy', 'Engineering', 'Materials Science'}),
    ('光学', {'Physics and Astronomy', 'Engineering'}),
    ('通信', {'Engineering', 'Computer Science'}),
    ('信息', {'Engineering', 'Computer Science', 'Physics and Astronomy'}),
    ('自动化', {'Engineering', 'Computer Science'}),
    ('电气', {'Engineering'}),
    ('机械', {'Engineering', 'Materials Science'}),
    ('制造', {'Engineering', 'Materials Science'}),
    ('机器人', {'Engineering', 'Computer Science'}),
    ('航空', {'Engineering', 'Physics and Astronomy'}),
    ('航天', {'Engineering', 'Physics and Astronomy'}),
    ('车辆', {'Engineering'}),
    ('动力', {'Engineering', 'Energy'}),
    ('能源', {'Energy', 'Engineering', 'Materials Science'}),
    ('土木', {'Engineering'}),
    ('建筑', {'Engineering'}),
    ('建工', {'Engineering'}),
    ('水利', {'Engineering', 'Environmental Science'}),
    ('交通', {'Engineering'}),
    ('船舶', {'Engineering'}),
    ('矿', {'Engineering', 'Earth and Planetary Sciences'}),
    ('冶金', {'Materials Science', 'Engineering'}),
    ('纺织', {'Materials Science', 'Engineering'}),
    ('环境', {'Environmental Science', 'Engineering', 'Earth and Planetary Sciences'}),
    ('生态', {'Environmental Science', 'Agricultural and Biological Sciences'}),
    ('地理', {'Earth and Planetary Sciences', 'Environmental Science', 'Social Sciences'}),
    ('地质', {'Earth and Planetary Sciences', 'Environmental Science'}),
    ('地球', {'Earth and Planetary Sciences', 'Environmental Science'}),
    ('大气', {'Earth and Planetary Sciences', 'Environmental Science'}),
    ('海洋', {'Earth and Planetary Sciences', 'Environmental Science', 'Agricultural and Biological Sciences'}),
    ('资源', {'Earth and Planetary Sciences', 'Environmental Science'}),
    ('基础医学', {'Medicine', 'Biochemistry, Genetics and Molecular Biology', 'Immunology and Microbiology', 'Neuroscience'}),
    ('口腔', {'Medicine', 'Dentistry'}),
    ('护理', {'Nursing', 'Medicine'}),
    ('公共卫生', {'Medicine', 'Environmental Science'}),
    ('药', {'Pharmacology, Toxicology and Pharmaceutics', 'Chemistry', 'Medicine', 'Biochemistry, Genetics and Molecular Biology'}),
    ('医院', {'Medicine', 'Biochemistry, Genetics and Molecular Biology', 'Immunology and Microbiology', 'Neuroscience'}),
    ('医学', {'Medicine', 'Biochemistry, Genetics and Molecular Biology', 'Immunology and Microbiology'}),
    ('医', {'Medicine', 'Biochemistry, Genetics and Molecular Biology', 'Immunology and Microbiology'}),
    ('脑', {'Neuroscience', 'Medicine', 'Psychology'}),
    ('神经', {'Neuroscience', 'Medicine'}),
    ('心理', {'Psychology', 'Neuroscience', 'Social Sciences'}),
    ('生命', {'Biochemistry, Genetics and Molecular Biology', 'Agricultural and Biological Sciences', 'Immunology and Microbiology'}),
    ('生物', {'Biochemistry, Genetics and Molecular Biology', 'Agricultural and Biological Sciences', 'Immunology and Microbiology'}),
    ('生科', {'Biochemistry, Genetics and Molecular Biology', 'Agricultural and Biological Sciences'}),
    ('农', {'Agricultural and Biological Sciences', 'Environmental Science', 'Biochemistry, Genetics and Molecular Biology'}),
    ('林', {'Agricultural and Biological Sciences', 'Environmental Science'}),
    ('植物', {'Agricultural and Biological Sciences', 'Biochemistry, Genetics and Molecular Biology'}),
    ('动物', {'Agricultural and Biological Sciences', 'Biochemistry, Genetics and Molecular Biology', 'Veterinary'}),
    ('兽医', {'Veterinary', 'Agricultural and Biological Sciences'}),
    ('食品', {'Agricultural and Biological Sciences', 'Chemistry', 'Chemical Engineering'}),
    ('园艺', {'Agricultural and Biological Sciences'}),
    ('经济', {'Economics, Econometrics and Finance', 'Business, Management and Accounting', 'Social Sciences'}),
    ('金融', {'Economics, Econometrics and Finance', 'Business, Management and Accounting'}),
    ('管理', {'Business, Management and Accounting', 'Decision Sciences', 'Economics, Econometrics and Finance', 'Social Sciences'}),
    ('商', {'Business, Management and Accounting', 'Economics, Econometrics and Finance'}),
    ('会计', {'Business, Management and Accounting', 'Economics, Econometrics and Finance'}),
    ('法', {'Social Sciences'}),
    ('社会', {'Social Sciences'}),
    ('教育', {'Social Sciences', 'Psychology'}),
]


def expected_fields(dept, org=''):
    s = (dept or '') + (org or '')
    for kw, fields in DEPT_FIELDS:
        if kw in s:
            return fields
    return None


def cand_fields(c):
    fs = set()
    for t in (c.get('topics') or [])[:4]:
        f = (t.get('field') or {}).get('display_name')
        if f:
            fs.add(f)
    return fs


def name_match(dn, given, surname):
    dn2 = dn.lower().replace('.', '').replace('-', '').replace(',', ' ')
    toks = [t for t in dn2.split() if t]
    if not toks:
        return False
    s, g = surname.lower(), given.lower()
    sur_ok = s in toks
    giv_ok = any(t == g or (len(t) > 1 and (t.startswith(g) or g.startswith(t))) for t in toks)
    return sur_ok and giv_ok


def find_author(name_cn, org_cn, dept=''):
    given, surname, py = pinyin_forms(name_cn)
    if not given:
        return {'err': 'bad_name'}
    iid = inst_id(org_cn)
    if not iid:
        return {'err': 'no_inst'}
    search = f'{given} {surname}'
    r = fetch(f'{API}/authors?filter=affiliations.institution.id:{iid[0]}&search={q(search)}'
              f'&per_page=25&select=id,display_name,works_count,cited_by_count,topics')
    matched = [c for c in r.get('results', []) if name_match(c['display_name'], given, surname)]
    if not matched:
        return {'err': 'no_match', 'inst': iid, 'search': search}
    exp = expected_fields(dept, org_cn)
    pool = matched
    field_gated = False
    if exp:
        fg = [c for c in matched if cand_fields(c) & exp]
        if fg:
            pool = fg
            field_gated = True
    best = max(pool, key=lambda c: c['cited_by_count'])
    # confidence
    others = [c for c in pool if c['id'] != best['id']]
    dominant = (not others) or (best['cited_by_count'] >= 3 * max([c['cited_by_count'] for c in others] + [1]))
    if len(matched) == 1:
        conf = 'high'
    elif field_gated and (len(pool) == 1 or dominant):
        conf = 'high'
    elif field_gated:
        conf = 'medium'
    elif dominant:
        conf = 'medium'
    else:
        conf = 'ambiguous'
    return {'author': best, 'inst': iid, 'search': search, 'conf': conf,
            'nmatch': len(matched), 'npool': len(pool), 'field_gated': field_gated}


def nature_papers(author_id, nat, natset):
    papers = []
    cur = '*'
    while cur:
        u = (f'{API}/works?filter=author.id:{author_id}&per_page=200&cursor={q(cur)}'
             f'&select=id,doi,title,publication_year,primary_location')
        r = fetch(u)
        for w in r.get('results', []):
            pl = w.get('primary_location') or {}
            src = (pl.get('source') or {})
            sid = (src.get('id') or '').split('/')[-1]
            if sid in natset:
                papers.append({'title': w.get('title'), 'year': w.get('publication_year'),
                               'journal': nat[sid], 'doi': w.get('doi')})
        cur = r['meta'].get('next_cursor')
        time.sleep(0.15)
    papers.sort(key=lambda p: (-(p['year'] or 0)))
    return papers


# ----------------------------------------------------------------------------
# Step 3: ORCID identity verification
# ----------------------------------------------------------------------------
def pyset(name_cn):
    py = lazy_pinyin(name_cn)
    return py[0].lower(), ''.join(py[1:]).lower()


def orcid_verify(matched_path):
    M = json.load(open(matched_path, encoding='utf-8'))
    n_checked = 0
    for r in M:
        r['orcid'] = None
        r['orcid_name'] = None
        r['orcid_name_ok'] = None
        if r.get('conf') not in ('high', 'medium'):
            continue
        aid = r.get('author', {}).get('id')
        if not aid:
            continue
        a = jget(f'https://api.openalex.org/authors/{aid}?select=id,orcid,display_name')
        time.sleep(0.08)
        orc = (a or {}).get('orcid')
        if not orc:
            continue
        oid = orc.split('/')[-1]
        r['orcid'] = oid
        p = jget(f'https://pub.orcid.org/v3.0/{oid}/person', 'application/json')
        if p and p.get('name'):
            gn = ((p['name'].get('given-names') or {}) or {}).get('value', '') or ''
            fn = ((p['name'].get('family-name') or {}) or {}).get('value', '') or ''
            r['orcid_name'] = (gn + ' ' + fn).strip()
            sur, giv = pyset(r['name'])
            full = (gn + fn).lower().replace('-', '').replace(' ', '')
            fnl = fn.lower().replace('-', '')
            gnl = gn.lower().replace('-', '')
            sur_ok = sur in fnl or fnl in sur or sur in full
            giv_ok = (giv[:4] in full) or (gnl and (gnl in giv or giv in gnl))
            r['orcid_name_ok'] = bool(sur_ok and giv_ok)
        n_checked += 1
        if n_checked % 20 == 0:
            json.dump(M, open(matched_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(n_checked, 'orcid-checked', flush=True)
        time.sleep(0.05)

    json.dump(M, open(matched_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    resolved = [r for r in M if r.get('conf') in ('high', 'medium')]
    has_orcid = [r for r in resolved if r.get('orcid')]
    name_ok = [r for r in has_orcid if r.get('orcid_name_ok')]
    name_bad = [r for r in has_orcid if r.get('orcid_name_ok') is False]
    print('=== ORCID SUMMARY ===')
    print('resolved:', len(resolved), '| has ORCID:', len(has_orcid),
          '| name-consistent:', len(name_ok), '| name-MISMATCH:', len(name_bad))
    suspect = [r for r in resolved if r['conf'] == 'medium' and r.get('papers') and
               (not r.get('orcid') or r.get('orcid_name_ok') is False or len(r['papers']) >= 25)]
    print('suspect (medium, review-priority):', len(suspect))
    for r in sorted(suspect, key=lambda x: -len(x['papers']))[:15]:
        print('  ', r['name'], r['org'], '| papers', len(r['papers']),
              '| orcid', r.get('orcid') or 'NONE', '| name_ok', r.get('orcid_name_ok'))


# ----------------------------------------------------------------------------
# Step 4: per-paper institution gate for MEDIUM matches
# ----------------------------------------------------------------------------
def inst_nature_dois(author_id, inst_oa_id, natset):
    """该作者【在那篇论文上确实挂靠 inst_oa_id】的 Nature 论文 DOI 集合。"""
    dois = set()
    cur = '*'
    while cur:
        u = (f'{API}/works?filter=author.id:{author_id},authorships.institutions.id:{inst_oa_id}'
             f'&per_page=200&cursor={q(cur)}&select=id,doi,primary_location')
        r = fetch(u)
        for w in r.get('results', []):
            sid = (((w.get('primary_location') or {}).get('source') or {}).get('id') or '').split('/')[-1]
            if sid in natset and w.get('doi'):
                dois.add(w['doi'])
        cur = (r.get('meta') or {}).get('next_cursor')
        time.sleep(0.1)
    return dois


def institution_gate(matched_path, natset):
    recs = json.load(open(matched_path, encoding='utf-8'))
    n = 0
    for r in recs:
        if r.get('conf') == 'high':
            for p in r.get('papers', []):
                p['inst_ok'] = True
        elif r.get('conf') == 'medium' and r.get('papers'):
            iid = inst_id(r['org'])
            oa = iid[0] if iid else None
            aid = r['author']['id']
            ok_dois = inst_nature_dois(aid, oa, natset) if oa else set()
            for p in r['papers']:
                p['inst_ok'] = bool(p.get('doi') and p['doi'] in ok_dois)
            n += 1
            if n % 10 == 0:
                json.dump(recs, open(matched_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                print(n, 'medium re-gated', flush=True)
            time.sleep(0.05)

    json.dump(recs, open(matched_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    hi = [r for r in recs if r.get('conf') == 'high' and r.get('papers')]
    md = [r for r in recs if r.get('conf') == 'medium' and r.get('papers')]
    hi_p = sum(len(r['papers']) for r in hi)
    md_ok = sum(sum(1 for p in r['papers'] if p.get('inst_ok')) for r in md)
    md_no = sum(sum(1 for p in r['papers'] if not p.get('inst_ok')) for r in md)
    md_ok_people = sum(1 for r in md if any(p.get('inst_ok') for p in r['papers']))
    print('=== GATE SUMMARY ===')
    print('HIGH people:', len(hi), '| HIGH papers (confirmed):', hi_p)
    print('MEDIUM papers inst-matched (confirmed):', md_ok, 'from', md_ok_people, 'people')
    print('MEDIUM papers NOT matched (uncertain):', md_no)
    print('confirmed total:', hi_p + md_ok, '| uncertain total:', md_no)


# ----------------------------------------------------------------------------
# match driver: read scholar list -> resolve authors + papers -> matched.json
# ----------------------------------------------------------------------------
def read_scholars(path):
    """读取 name,org[,dept] 列的 CSV/TSV(首行可为表头)。"""
    rows = []
    delim = '\t' if path.lower().endswith(('.tsv', '.txt')) else ','
    with open(path, encoding='utf-8') as f:
        for cols in csv.reader(f, delimiter=delim):
            cols = [c.strip() for c in cols if c is not None]
            if not cols or not cols[0]:
                continue
            if cols[0] in ('name', '姓名'):     # 跳过表头
                continue
            name = cols[0]
            org = cols[1] if len(cols) > 1 else ''
            dept = cols[2] if len(cols) > 2 else ''
            rows.append((name, org, dept))
    return rows


def run_match(input_path, sources_path, out_path):
    nat, natset = load_sources(sources_path)
    scholars = read_scholars(input_path)
    out = []
    for name, org, dept in scholars:
        m = find_author(name, org, dept)
        rec = {'name': name, 'org': org, 'dept': dept}
        if not m or 'err' in m:
            rec['err'] = (m or {}).get('err', 'unknown')
            out.append(rec)
            print(f'{name} ({org}): MISS {rec["err"]}')
            continue
        a = m['author']
        ps = nature_papers(a['id'].split('/')[-1], nat, natset)
        rec.update({'author': a, 'inst': m['inst'], 'conf': m['conf'],
                    'nmatch': m['nmatch'], 'npool': m['npool'],
                    'field_gated': m['field_gated'], 'papers': ps})
        out.append(rec)
        print(f"{name} ({org}) -> {a['display_name']} | works={a['works_count']} "
              f"cited={a['cited_by_count']} | conf={m['conf']} | NaturePapers={len(ps)}")
        time.sleep(0.3)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\nwritten {out_path} | {len(out)} scholars')


def main():
    ap = argparse.ArgumentParser(description='Match Chinese scholars to OpenAlex authors + Nature papers.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    s1 = sub.add_parser('build-sources', help='build the Nature-family source set')
    s1.add_argument('--sources', default='./nature_sources.json')

    s2 = sub.add_parser('match', help='match scholars from CSV/TSV -> matched.json')
    s2.add_argument('--input', required=True, help='CSV/TSV with columns name,org[,dept]')
    s2.add_argument('--sources', default='./nature_sources.json')
    s2.add_argument('--out', default='./matched.json')

    s3 = sub.add_parser('orcid', help='ORCID-verify matched authors (in place)')
    s3.add_argument('--matched', default='./matched.json')

    s4 = sub.add_parser('gate', help='per-paper institution gate for medium matches (in place)')
    s4.add_argument('--matched', default='./matched.json')
    s4.add_argument('--sources', default='./nature_sources.json')

    args = ap.parse_args()
    if args.cmd == 'build-sources':
        ids = build_nature_sources(args.sources)
        print('nature-family sources:', len(ids))
    elif args.cmd == 'match':
        run_match(args.input, args.sources, args.out)
    elif args.cmd == 'orcid':
        orcid_verify(args.matched)
    elif args.cmd == 'gate':
        _, natset = load_sources(args.sources)
        institution_gate(args.matched, natset)


if __name__ == '__main__':
    main()
