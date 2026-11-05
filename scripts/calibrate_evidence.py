"""从已保存的Ch04真实正式排名校准；明确记录历史来源，不冒充新检索。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.learning.confidence import calibrate


def main(source):
    path = Path(source)
    data = json.loads(path.read_text(encoding='utf-8'))
    cases = [r for r in data['results'] if r['strategy'] == 'hybrid_rerank']
    dataset = ROOT / 'tests/fixtures/ch04_eval.json'
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != data['dataset_sha256'] or len(cases) != 50:
        raise ValueError('必须使用同一冻结Ch04集的完整50题报告')
    profile, report = calibrate(cases)
    profile.update(
        dataset_sha256=digest,
        source_report=str(path),
        source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_code_sha256=data['code_sha256'],
        corpus=data['corpus'],
    )
    report.update(profile=profile, data_origin='历史真实检索快照，本次仅离线校准')
    (ROOT / 'config/evidence-confidence.json').write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (ROOT / 'dev-notes/ch09-calibration.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(
        json.dumps(
            {k: report[k] for k in ('train', 'validation', 'limitation')},
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    main(parser.parse_args().report)
