# -*- coding: utf-8 -*-
"""Worker: audit a single paper directory, emit a JSON record to stdout.

Designed to be called by a batch driver via subprocess.run(timeout=...), so that
a pathological file (e.g. an openpyxl parse that hangs) only stalls this child
process and gets killed by the parent's timeout, instead of freezing the batch.

argv[1] = directory path.
Output: {slug, hard, n_all, n_xlsx, oversized_skipped, err?}
  - n_all: total candidate flags
  - hard:  the high-confidence detector families (A/G/F/H)
  - oversized_skipped: files skipped for exceeding MAXMB
"""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forensic

HARD = ('A 两列恒差', 'G 块级小数锁定', 'F 同图内重复块', 'H 行级重复块')
MAXMB = 45


def main():
    d = sys.argv[1]
    slug = os.path.basename(os.path.normpath(d))
    xlsx = glob.glob(d.rstrip('/\\') + os.sep + '*.xlsx')
    rec = {'slug': slug, 'n_xlsx': 0, 'n_all': 0, 'hard': []}
    oversized = [(os.path.basename(p), round(os.path.getsize(p) / 1e6, 1))
                 for p in xlsx if os.path.getsize(p) > MAXMB * 1e6]
    if oversized:
        rec['oversized_skipped'] = oversized
    xlsx = [p for p in xlsx if os.path.getsize(p) <= MAXMB * 1e6]
    rec['n_xlsx'] = len(xlsx)
    if xlsx:
        try:
            flags = forensic.audit_paths(xlsx)
            rec['n_all'] = len(flags)
            rec['hard'] = [{'tag': f['tag'], 'sheet': f['sheet'], 'msg': f['msg']}
                           for f in flags if f['tag'] in HARD]
        except Exception as e:
            rec['err'] = repr(e)
    sys.stdout.write(json.dumps(rec, ensure_ascii=False))


if __name__ == '__main__':
    main()
