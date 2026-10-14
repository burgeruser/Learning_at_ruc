# -*- coding: utf-8 -*-
"""Extract high-resolution figure images from already-fetched paper HTML.

Reads each paper's fulltext.html (produced by fetch_sources.py), extracts the
figure media URLs (main-text figures + supplementary ESM images), normalizes
them to lw1200 (1200px high-res), and downloads them into <paper>/figs/.

Resumable (skips already-downloaded files). The downloaded figures feed
image_forensic.py for ORB feature comparison.

Proxy: inherited from fetch_sources (reads GENGSCAN_PROXY env var).

Usage:
    python fetch_figures.py [--indir ./papers] [--limit N] [--workers 5]
"""
import re, os, glob, sys, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_sources


def figs_from_html(h):
    """提取 MediaObjects 下的 figure 图,统一为 lw1200 高清。主文 _Fig\\d_HTML + SI _Fig\\d_ESM。"""
    urls = set()
    for m in re.finditer(
            r'springer-static/image/(art%3A[^"\'\s/]+/MediaObjects/[A-Za-z0-9_%]+_Fig\d+_(?:HTML|ESM)\.(?:png|jpg|jpeg))', h):
        urls.add('https://media.springernature.com/lw1200/springer-static/image/' + m.group(1))
    return sorted(urls)


def fetch_one(paper_dir):
    fp = os.path.join(paper_dir, 'fulltext.html')
    slug = os.path.basename(os.path.normpath(paper_dir))
    if not os.path.exists(fp):
        return {'slug': slug, 'n_urls': 0, 'got': 0}
    h = open(fp, encoding='utf-8', errors='ignore').read()
    urls = figs_from_html(h)
    outd = os.path.join(paper_dir, 'figs')
    os.makedirs(outd, exist_ok=True)
    got = 0
    for u in urls:
        fn = u.split('/')[-1]
        p = os.path.join(outd, fn)
        if os.path.exists(p) and os.path.getsize(p) > 2000:
            got += 1; continue
        for a in range(2):
            fetch_sources.curl(u, out=p, proxy=(a == 0))
            if os.path.exists(p) and os.path.getsize(p) > 2000 and open(p, 'rb').read(1) != b'<':
                got += 1; break
    return {'slug': slug, 'n_urls': len(urls), 'got': got}


def main():
    ap = argparse.ArgumentParser(description='Extract high-res figures from fetched paper HTML.')
    ap.add_argument('--indir', default='./papers', help='root dir holding <paper>/fulltext.html')
    ap.add_argument('--limit', type=int, default=None, help='only process first N papers')
    ap.add_argument('--workers', type=int, default=5, help='concurrent download workers')
    args = ap.parse_args()

    # 所有含 fulltext.html 的论文目录
    dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(args.indir, '*', 'fulltext.html'))]
    dirs.sort()
    if args.limit:
        dirs = dirs[:args.limit]
    print(f'抓图目标 {len(dirs)} 篇 | 并发 {args.workers}', flush=True)

    tot_fig = with_fig = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, d): d for d in dirs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            tot_fig += rec['got']
            if rec['got'] > 0:
                with_fig += 1
            if i % 25 == 0:
                print(f'[{i}/{len(dirs)}] 累计下图 {tot_fig} 张 / {with_fig} 篇有图', flush=True)
    print(f'\n完成。{len(dirs)} 篇 | 下图 {tot_fig} 张 | {with_fig} 篇有图 → figs/', flush=True)


if __name__ == '__main__':
    main()
