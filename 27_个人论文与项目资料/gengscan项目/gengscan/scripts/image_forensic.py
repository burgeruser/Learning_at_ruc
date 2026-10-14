# -*- coding: utf-8 -*-
"""Image forensics via ORB local-feature matching (+ copy-move detection).

Complements perceptual-hash methods (which miss partial/transformed copies) while
reducing false positives from generic scientific-figure elements (axes, schematics):
 1. Inlier-ratio threshold: inliers/keypoints >= FRAC. Real copies form a high-ratio
    contiguous block; generic figure elements match at <5%.
 2. "Star image" filter: a figure matching > MAXHIT papers is a generic element
    (schematic / axis), so all of its candidates are removed.
 3. Copy-move (within a single image): far-apart self-matching points that are
    geometrically consistent (RANSAC) reveal a duplicated block within one image
    (e.g. a self-copied western-blot lane) that cross-image and pHash both miss.

A match is NOT proof of fabrication; it produces candidates for VLM/human review.

Usage:
    python image_forensic.py [--indir ./papers] [--out image_forensic_report.txt]

Scans <indir>/*/figs/*.png|jpg|jpeg by default.
"""
import cv2, numpy as np, glob, os, sys, argparse
from collections import Counter

ORB = cv2.ORB_create(nfeatures=2000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING)
MAXDIM = 1600
MIN_INLIERS = 30
FRAC = 0.08               # inlier 占比下限(真复制高占比)
MAXHIT = 2                # 一张图匹配 >此数篇 = 通用图元,剔除
RATIO = 0.75
CM_MIN = 15               # copy-move 几何一致点下限


def load(path):
    data = np.fromfile(path, dtype=np.uint8)        # 绕过 cv2 中文路径限制
    im = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None, None
    h, w = im.shape
    if min(h, w) < 80:
        return None, None, None
    s = min(1.0, MAXDIM / max(h, w))
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)))
    kp, des = ORB.detectAndCompute(im, None)
    return im, kp, des


def match(a, b):
    kp1, des1 = a; kp2, des2 = b
    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return 0, 0.0
    try:
        knn = BF.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0, 0.0
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < RATIO * n.distance]
    if len(good) < 15:
        return len(good), 0.0
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, inliers / max(len(kp1), len(kp2))


def copymove(kp, des):
    """同图内 copy-move:远距离自匹配点 + 几何一致(RANSAC)=重复块。"""
    if des is None or len(kp) < 60:
        return 0
    try:
        knn = BF.knnMatch(des, des, k=3)
    except cv2.error:
        return 0
    p1s, p2s = [], []
    for matches in knn:
        cand = [m for m in matches if m.queryIdx != m.trainIdx]
        if len(cand) >= 2 and cand[0].distance < RATIO * cand[1].distance:
            m = cand[0]
            a = kp[m.queryIdx].pt; b = kp[m.trainIdx].pt
            if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 > 40:   # 排除邻域
                p1s.append(a); p2s.append(b)
    if len(p1s) < CM_MIN:
        return 0
    src = np.float32(p1s).reshape(-1, 1, 2); dst = np.float32(p2s).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def slug_of_path(p):
    """以图片所在论文目录名作为分组标识(<indir>/<slug>/figs/img.png → slug)。"""
    q = p.replace('\\', '/').split('/')
    if 'figs' in q:
        i = q.index('figs')
        if i >= 1:
            return q[i-1]
    return q[-1]


def main():
    ap = argparse.ArgumentParser(description='ORB image forensics: copy-move + panel reuse.')
    ap.add_argument('--indir', default='./papers', help='root dir holding <paper>/figs/*.png')
    ap.add_argument('--out', default='./image_forensic_report.txt', help='output report path')
    args = ap.parse_args()

    feats = {}
    for p in glob.glob(os.path.join(args.indir, '*', 'figs', '*')):
        if not p.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        im, kp, des = load(p)
        if des is not None and len(kp) >= 15:
            feats[p] = (kp, des, slug_of_path(p))
    print(f'载入 {len(feats)} 张可比对图', flush=True)

    # 同图 copy-move
    cmove = []
    for p, (kp, des, _) in feats.items():
        n = copymove(kp, des)
        if n >= CM_MIN:
            cmove.append((n, p))

    # 图间比对(占比阈值)
    keys = list(feats.keys())
    raw_w, raw_c = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pa, pb = keys[i], keys[j]
            ka, da, sa = feats[pa]; kb, db, sb = feats[pb]
            if os.path.basename(pa) == os.path.basename(pb) and sa == sb:
                continue
            inl, frac = match((ka, da), (kb, db))
            if inl >= MIN_INLIERS and frac >= FRAC:
                (raw_w if sa == sb else raw_c).append((inl, round(frac, 3), pa, pb))

    # 排明星图(通用图元):一张图配 >MAXHIT 篇
    cnt = Counter()
    for inl, frac, pa, pb in raw_w + raw_c:
        cnt[pa] += 1; cnt[pb] += 1
    star = {p for p, c in cnt.items() if c > MAXHIT}
    within = [x for x in raw_w if x[2] not in star and x[3] not in star]
    cross = [x for x in raw_c if x[2] not in star and x[3] not in star]

    rep = open(args.out, 'w', encoding='utf-8')
    rep.write(f'ORB(占比>={FRAC} + 排明星图 + copy-move)。重复≠造假,供VLM/人核。\n')
    rep.write(f'载入 {len(feats)} 张图。剔除通用图元(明星图) {len(star)} 张。\n')
    rep.write(f'\n===== 同图 copy-move(自我复制块) ({len(cmove)}) =====\n')
    for n, p in sorted(cmove, reverse=True):
        rep.write(f'  几何一致重复点 {n} | {p.split("/")[-1]} ({slug_of_path(p)})\n')
    for title, lst in [('篇内 panel 复用', within), ('跨篇复用', cross)]:
        rep.write(f'\n===== {title} ({len(lst)}) =====\n')
        for inl, frac, pa, pb in sorted(lst, reverse=True):
            rep.write(f'  inlier {inl} (占比{frac}) | {pa.split("/")[-1]} <=> {pb.split("/")[-1]}\n')
    rep.close()
    print(f'copy-move {len(cmove)} | 篇内 {len(within)} | 跨篇 {len(cross)} (剔明星图{len(star)}) '
          f'→ {args.out}', flush=True)


if __name__ == '__main__':
    main()
