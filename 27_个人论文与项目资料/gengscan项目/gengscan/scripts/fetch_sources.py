# -*- coding: utf-8 -*-
"""Robustly fetch ALL Source Data files for a paper (one MOESM per figure).

Handles common access obstacles:
 1. Some publisher pages are behind anti-scraping walls (direct curl returns 0 bytes)
    -> tries the proxy first (if configured via GENGSCAN_PROXY), direct as fallback.
 2. Source Data is not only .xlsx: recent papers bundle MOESM_ESM.zip, or use
    .xls/.csv -> regex covers all extensions, archives are unzipped to extract tables.
 3. Data hosted in external repositories (figshare/zenodo/dryad) -> the links are
    recorded for human review (not auto-downloaded; their formats vary widely).
 4. Distinguishes 'fetch failed (empty page)' vs 'page fetched but no data links'
    via a status field, instead of recording n_links=0 for both.

Proxy: read from environment variable GENGSCAN_PROXY (e.g. http://HOST:PORT)
or pass --proxy. Defaults to no proxy.

Usage:
    python fetch_sources.py slug1=DOI_OR_URL slug2=DOI_OR_URL ...
    python fetch_sources.py --outdir ./papers --proxy http://HOST:PORT slug=DOI
"""
import subprocess, re, os, time, sys, zipfile, glob, argparse

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
PROXY = os.environ.get('GENGSCAN_PROXY')   # None = 不走代理


def curl(url, out=None, proxy=False, timeout=120):
    cmd = ['curl', '-sL', '--max-time', str(timeout), '-A', UA, '-H', 'Referer: https://www.nature.com/']
    cmd += (['-x', PROXY] if (proxy and PROXY) else ['--noproxy', '*'])
    if out:
        cmd += [url, '-o', out]; subprocess.run(cmd, capture_output=True); return None
    return subprocess.run(cmd + [url], capture_output=True, text=True, encoding='utf-8', errors='ignore').stdout


def good_dl(p):
    """下载有效:存在 + >800字节 + 不是反爬返回的 HTML(以 '<' 开头)。"""
    if not (os.path.exists(p) and os.path.getsize(p) > 800):
        return False
    return open(p, 'rb').read(1) != b'<'


def unzip_data(zip_path, out):
    """解压 MOESM zip,把内部表格(xlsx/csv/xls)提取到 out 目录(zip_ 前缀防冲突)。"""
    n = 0
    try:
        with zipfile.ZipFile(zip_path) as z:
            for nm in z.namelist():
                if nm.startswith('__') or '/__' in nm:
                    continue
                if not nm.lower().endswith(('.xlsx', '.csv', '.xls')):
                    continue
                if z.getinfo(nm).file_size > 2e8:   # >200MB 跳过
                    continue
                base = os.path.basename(nm)
                if not base:
                    continue
                with z.open(nm) as src, open(f'{out}/zip_{base}', 'wb') as dst:
                    dst.write(src.read())
                n += 1
    except Exception:
        pass
    return n


def fetch_all(slug, doi, outdir='./papers', tries=5):
    if 'nature.com/articles/' in doi:
        url = doi if doi.startswith('http') else 'https://' + doi[doi.index('nature.com'):]
    else:
        art = doi.split('doi.org/')[-1]
        url = 'https://www.nature.com/articles/' + (art.split('/', 1)[1] if art.startswith('10.1038/') else art)
    out = f'{outdir}/{slug}'
    os.makedirs(out, exist_ok=True)
    # 默认先代理(直连常被反爬),直连兜底
    h = curl(url, proxy=True) or curl(url) or ''
    open(f'{out}/fulltext.html', 'w', encoding='utf-8').write(h)
    if not h:
        return {'slug': slug, 'url': url, 'status': 'fetch_failed',
                'n_links': 0, 'n_got': 0, 'n_xlsx': 0, 'ext_repos': [], 'missing': []}
    # publisher ESM 数据文件(xlsx/xls/csv/zip)
    links = sorted(set(re.findall(
        r'https://static-content\.springer\.com/esm/[^"\'\s]+?\.(?:xlsx|xls|csv|zip)', h)))
    # 外部数据仓库链接(记录供人看,不自动下)
    ext = sorted(set(re.findall(r'https://[^"\'\s]*?(?:figshare|zenodo|dryad)[^"\'\s]*', h)))[:6]
    got = []
    for l in links:
        l = l.replace('&amp;', '&')
        fn = l.split('/')[-1]
        p = f'{out}/{fn}'
        if not good_dl(p):
            for attempt in range(tries):
                curl(l, out=p, proxy=(attempt % 2 == 0))   # 默认代理优先
                if good_dl(p):
                    break
                time.sleep(1.5)
        if good_dl(p):
            got.append(fn)
            if fn.lower().endswith('.zip'):
                unzip_data(p, out)
    n_xlsx = len(glob.glob(f'{out}/*.xlsx'))
    status = 'ok' if (links or ext) else 'page_no_data'   # 拿到页面但无数据链接(可能真无/纯PDF SI)
    return {'slug': slug, 'url': url, 'status': status, 'n_links': len(links),
            'n_got': len(got), 'n_xlsx': n_xlsx, 'ext_repos': ext,
            'missing': [l.split('/')[-1] for l in links if l.split('/')[-1] not in got]}


def main():
    global PROXY
    ap = argparse.ArgumentParser(description='Fetch all Source Data files for papers.')
    ap.add_argument('pairs', nargs='*', help='slug=DOI_OR_URL entries')
    ap.add_argument('--outdir', default='./papers', help='output root dir for fetched papers')
    ap.add_argument('--proxy', default=None, help='HTTP proxy (overrides GENGSCAN_PROXY env)')
    args = ap.parse_args()
    if args.proxy:
        PROXY = args.proxy

    pairs = []
    for a in args.pairs:
        s, d = a.split('=', 1)
        pairs.append((s, d))
    if not pairs:
        print('usage: python fetch_sources.py slug=DOI_OR_URL [...]')
        return
    for s, d in pairs:
        r = fetch_all(s, d, outdir=args.outdir)
        print(f"[{r['slug']}] {r['status']} | esm链接 {r['n_links']} | 下到 {r['n_got']}"
              f" | 现有xlsx {r['n_xlsx']} | 外链 {len(r['ext_repos'])}"
              + (f" {r['ext_repos'][:1]}" if r['ext_repos'] else ''))


if __name__ == '__main__':
    main()
