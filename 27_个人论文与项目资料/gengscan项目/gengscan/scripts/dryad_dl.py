# -*- coding: utf-8 -*-
"""Download a public Dryad file_stream past the Anubis proof-of-work wall.

Anubis "fast" algorithm: sha256_hex(randomData + nonce); find a nonce whose hash
has `difficulty` leading '0' chars, then request pass-challenge to obtain an auth
cookie, and re-fetch the file with that cookie. Difficulty 4 solves in ~seconds.

This targets openly-licensed public research data; clearing a rate-limit PoW is
not a circumvention of access control.

Usage:
    python dryad_dl.py <file_stream_url> <out_path> [--jar cookies.txt]
"""
import sys, re, json, hashlib, subprocess, os, time, argparse

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
JAR = './.dryad_cookies.txt'


def curl(args):
    return subprocess.run(['curl', '-sL', '--max-time', '120', '-A', UA,
                           '-c', JAR, '-b', JAR] + args,
                          capture_output=True, text=True, encoding='utf-8', errors='ignore').stdout


def solve(random_data, difficulty):
    target = '0' * difficulty
    nonce = 0
    while True:
        h = hashlib.sha256((random_data + str(nonce)).encode()).hexdigest()
        if h.startswith(target):
            return h, nonce
        nonce += 1


def download(url, out, dbg=False):
    from urllib.parse import quote
    os.makedirs(os.path.dirname(JAR) or '.', exist_ok=True)
    if os.path.exists(JAR):
        os.remove(JAR)
    page = curl([url])
    m = re.search(r'id="anubis_challenge"[^>]*>(.*?)</script>', page, re.S)
    if not m:
        if page and not page.lstrip().lower().startswith(('<!doctype', '<html')):
            open(out, 'w', encoding='utf-8', errors='ignore').write(page)
            return os.path.exists(out) and os.path.getsize(out) > 200
        if dbg: print('  no challenge found; page head:', page[:120], file=sys.stderr)
        return False
    ch = json.loads(m.group(1))
    rd = ch['challenge']['randomData']
    diff = ch['challenge']['difficulty']
    cid = ch['challenge']['id']
    h, nonce = solve(rd, diff)
    if dbg: print(f'  solved diff={diff} nonce={nonce} hash={h[:12]}..', file=sys.stderr)
    pc = ('https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge'
          f'?id={quote(cid)}&response={h}&nonce={nonce}&redir={quote(url, safe="")}&elapsedTime=2200')
    # 直接跟随 pass-challenge 的重定向链取文件(会话连续,避免负载均衡重取落到无session后端)
    subprocess.run(['curl', '-sL', '--max-time', '120', '-A', UA, '-c', JAR, '-b', JAR, pc, '-o', out],
                   capture_output=True)
    if os.path.exists(out) and os.path.getsize(out) > 200:
        head = open(out, 'rb').read(64).lstrip().lower()
        if not (head.startswith(b'<!doctype') or head.startswith(b'<html') or b'<title>' in head):
            return True
        if dbg: print('  still html after pass:', head[:60], file=sys.stderr)
    elif dbg:
        print('  empty/small after pass, size:', os.path.getsize(out) if os.path.exists(out) else 'none', file=sys.stderr)
    return False


def download_retry(url, out, tries=5, dbg=False):
    """Anubis + 负载均衡粘性会偶发抖动(空页/挑战重出),重试至成功。"""
    for i in range(tries):
        if download(url, out, dbg=dbg):
            return True
        time.sleep(1.2)
    return False


def main():
    global JAR
    ap = argparse.ArgumentParser(description='Download a public Dryad file past Anubis PoW.')
    ap.add_argument('url', help='Dryad file_stream URL')
    ap.add_argument('out', help='output path')
    ap.add_argument('--jar', default=None, help='cookie jar path (default ./.dryad_cookies.txt)')
    ap.add_argument('--tries', type=int, default=6, help='retry attempts')
    args = ap.parse_args()
    if args.jar:
        JAR = args.jar
    ok = download_retry(args.url, args.out, tries=args.tries, dbg=True)
    print(json.dumps({'url': args.url, 'out': args.out, 'ok': ok,
                      'size': os.path.getsize(args.out) if os.path.exists(args.out) else 0}))


if __name__ == '__main__':
    main()
