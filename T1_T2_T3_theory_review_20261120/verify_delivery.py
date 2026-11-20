"""Verify reviewer archive, input hashes, source scope and mathematical checks."""
from hashlib import sha256
from pathlib import Path
import ast
import json
import re
import subprocess

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def main():
    manifest=json.loads((HERE/"input_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["attachments"]:
        path=HERE/entry["archived_file"]
        assert path.stat().st_size==entry["bytes"]
        assert sha256(path.read_bytes()).hexdigest()==entry["sha256"]
    result=json.loads((HERE/"checks/theory_checks.json").read_text(encoding="utf-8"))
    assert result["status"]=="passed"
    assert abs(result["constants"]["sigma_min_lower"]-.06592289483611202)<1e-13
    assert result["deterministic_grid"]["points"]==1440
    assert result["deterministic_grid"]["left_inverse_max_error"]<5e-7
    assert result["exact_boundary_and_compensation"]["onset_duration_pairs"]==130
    for file in HERE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"))
    links=0
    docs=list(HERE.glob("*.md"))+[BASE/"README.md",BASE/"research_discussions/README.md"]
    for file in docs:
        for match in re.finditer(r"\]\(([^)]+)\)",file.read_text(encoding="utf-8")):
            target=match.group(1).split("#")[0]
            if target.startswith(("http://","https://")) or not target:
                continue
            # Root historical README contains old links intentionally outside the
            # uploaded package. Verify all new-review links and all local package links.
            if file.parent!=HERE and "T1_T2_T3_theory_review_20261120" not in target:
                continue
            resolved=file.parent/target
            # This report is created by the present successful verification.
            if resolved.resolve()!=(HERE/"checks/delivery_checks.json").resolve():
                assert resolved.exists(),(str(file),target)
            links+=1
    # No tracked preexisting algorithm/model/data changes may be included.
    changes=subprocess.check_output(["git","diff","--name-only"],cwd=BASE,text=True).splitlines()
    allowed={"README.md","research_discussions/README.md"}
    assert all(name in allowed or name.startswith(HERE.name+"/") for name in changes),changes
    sources={name:sha256((BASE/name).read_bytes()).hexdigest() for name in (
        "common_g_reference_revision_20261115/common_g_reference.py",
        "common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json",
        "JFI_reconstruction_attack_aware_soft_accommodation_V1/accommodation.py",
        "B1_three_hypothesis_feasibility_20261120/registration.json")}
    for name in sources:
        expected=subprocess.check_output(["git","show","HEAD:"+name],cwd=BASE)
        working=(BASE/name).read_bytes()
        assert expected.replace(b"\r\n",b"\n")==working.replace(b"\r\n",b"\n"),name
    output=dict(status="passed",date="2026-11-20",perspective="reviewer",
        verified_original_attachments=len(manifest["attachments"]),relative_links=links,
        existing_sources_unchanged=sources,physical_paths_generated=0,
        deterministic_algebra_points=1440,scope="Formula and archive checks, not efficacy or novelty certification")
    (HERE/"checks/delivery_checks.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
