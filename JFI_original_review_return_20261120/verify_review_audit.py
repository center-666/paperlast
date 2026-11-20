"""Check the review ledger, canonical-source audit and exact timing example."""

import argparse
from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = "common_g_reference_revision_20261115/reviewer_status_update.json"
MATRIX = "02_29条意见与采用证据.md"
STATUSES = {"可采用":"已有可采用证据", "待补证":"仍需明确补充", "未关闭":"当前无法关闭"}
NOVELTY = {"R1.1", "R1.2", "R2.2", "R3.5", "R5.3"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def verify_review_text(source, rows, reg):
    data = source.read_bytes()
    assert sha256(data).hexdigest() == reg["private_review_attached_sha256"]
    lines = data.decode("utf-8-sig").splitlines()
    counts = Counter()
    last_heading = None
    position = {}
    for i, line in enumerate(lines):
        heading = re.match(r"^Reviewer #(\d+):", line)
        if heading:
            last_heading = int(heading.group(1))
        if line.startswith(("%ATTACH_", "***")):
            last_heading = None
        item = re.match(r"^(\d+)\.\s*(.*)", line)
        if last_heading and item:
            position[f"R{last_heading}.{item[1]}"] = i
            counts[last_heading] += 1
    assert [counts[k] for k in range(1, 7)] == reg["review_counts_original"]
    assert set(position) == {r["id"] for r in rows}
    for i, row in enumerate(rows):
        start = position[row["id"]]
        assert start+1 == row["source_start_line"], row["id"]
        begin = re.fullmatch(r"\d+\.\s*(.*)", lines[start]).group(1)
        stop = position[rows[i+1]["id"]] if i+1 < len(rows) else len(lines)
        body = [begin]
        for line in lines[start+1:stop]:
            if re.match(r"^Reviewer #\d+:", line) or line.startswith(("%ATTACH_", "***")):
                break
            body.append(line)
        assert norm(" ".join(body)) == norm(row["text"]), row["id"]
    return len(rows)


def verify_tex(path, reg):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == reg["local_retained_manuscript_tex_sha256"]
    anchors = {144: r"\tag{3}", 151: r"\tag{4}", 295: r"\tag{13}", 319: r"\tag{16}",
               472: r"\tag{30}", 492: r"\tag{34}", 584: r"\tag{39}", 618: r"\tag{42}",
               966: r"\tag{53}", 1111: r"\section{Simulation}"}
    lines = raw.decode("utf-8-sig").splitlines()
    for n, token in anchors.items():
        assert token in lines[n-1], (n, token)
    return len(anchors)


def scalar_exact_check():
    # D^2 = 2*d1^2 + 2*d2^2 + 2*d1*d2; the two diagonal precision
    # entries are both 2 and the off-diagonal entries are both 1.
    def distance(m, n):
        return F(2*m*m+2*n*n+2*m*n)
    atoms = {(m,n):distance(m,n) for m in (1,2) for n in (1,2)}
    low, mid, high = atoms[(1,1)], atoms[(1,2)], atoms[(2,2)]
    assert low == 6 and mid == atoms[(2,1)] == 14 and high == 24
    weights = {
        "frechet_contrast_positive": (F(1,2), F(0), F(0), F(1,2)),
        "independent_abstraction": (F(1,4),)*4,
        "frechet_contrast_negative": (F(0), F(1,2), F(1,2), F(0))}
    for w in weights.values():
        assert len(w) == 4 and sum(w) == 1
        assert w[0]+w[1] == w[0]+w[2] == F(1,2)
    means = {k:sum(wi*atoms[a] for a,wi in zip(atoms,w,strict=True)) for k,w in weights.items()}
    assert list(means.values()) == [F(15), F(29,2), F(14)]
    # The zero-cross-precision example has equal mean distances, but one
    # distance is constant and the other is not. Strict Jensen orders risk.
    diagonal = {a:F(2*a[0]**2+2*a[1]**2) for a in atoms}
    assert list(diagonal.values()) == [4,10,10,16]
    assert all(sum(wi*diagonal[a] for a,wi in zip(diagonal,w,strict=True))==10
               for w in weights.values())
    return {"nonzero_cross":{k:str(v) for k,v in means.items()},
            "zero_cross_all_equal":"10"}


def verify_public():
    reg = read(HERE/"registration.json")
    for name, expected in reg["published_dependencies"].items():
        p = BASE/name
        assert sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() == expected, name
    source = read(BASE/SOURCE)
    rows = source["rows"]
    assert len(rows) == len({x["id"] for x in rows}) == 29
    counts = Counter(r["reviewer"] for r in rows)
    assert [counts[x] for x in range(1,7)] == reg["review_counts_original"]

    matrix = (HERE/MATRIX).read_text(encoding="utf-8")
    matches = re.findall(r"^\|(R\d\.\d)\|(\d+)\|(可采用|待补证|未关闭)\|", matrix, flags=re.M)
    assert len(matches) == len({x[0] for x in matches}) == 29
    table = {i:(int(l), STATUSES[s]) for i,l,s in matches}
    for a in rows:
        assert table[a["id"]] == (a["source_start_line"],a["status"]), a["id"]
    assert {x["id"] for x in rows if x["status"]=="当前无法关闭"} == NOVELTY
    assert Counter(x[1] for x in table.values()) == source["counts"]

    assert set(reg["published_dependencies"]) >= {SOURCE}
    return reg, rows, {"review_items":29, "reviewer_item_counts":reg["review_counts_original"],
                       "statuses":dict(Counter(x[1] for x in table.values())),
                       "frozen_dependencies":len(reg["published_dependencies"]),
                       "exact_scalar_mean_squared_distances":scalar_exact_check()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-source", type=Path)
    parser.add_argument("--retained-tex", type=Path)
    parser.add_argument("--retained-pdf", type=Path)
    args = parser.parse_args()
    reg, rows, result = verify_public()
    if args.review_source:
        result["checked_private_review_body_count"] = verify_review_text(args.review_source, rows, reg)
    if args.retained_tex:
        result["verified_retained_tex_anchors"] = verify_tex(args.retained_tex, reg)
    if args.retained_pdf:
        assert sha256(args.retained_pdf.read_bytes()).hexdigest() == reg["local_retained_manuscript_pdf_sha256"]
        result["retained_pdf_hash_verified"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
