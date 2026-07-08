"""
provenance.py — 체크포인트/데이터 '내용' 해시 (B1/B3)

v2의 결함: 파일명+크기만 해시 → 모든 LoRA 변형이 같은 해시(존재 이유 무효).
v3: 실제 바이트를 해시한다.
  - 작은 파일(<64MB, adapter/config/json)은 전체 sha1
  - 큰 shard는 (크기 + 앞 4MB + 뒤 4MB) sha1 — 재학습/재저장 감지에 충분
"""

import hashlib
import json
from pathlib import Path

_SMALL = 64 * 1024 * 1024
_CHUNK = 4 * 1024 * 1024


def file_hash(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as f:
        if size <= _SMALL:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
        else:
            h.update(f.read(_CHUNK))
            f.seek(max(0, size - _CHUNK))
            h.update(f.read(_CHUNK))
    return h.hexdigest()


def dir_hash(path: Path, patterns=("*.safetensors", "*.bin", "adapter_*.json",
                                   "config.json")) -> str:
    """디렉토리 내용 해시. 파일 없으면 'no-ckpt'."""
    path = Path(path)
    if not path.exists():
        return "no-ckpt"
    files = sorted({f for pat in patterns for f in path.glob(pat)})
    if not files:
        return "no-ckpt"
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode())
        h.update(file_hash(f).encode())
    return h.hexdigest()[:16]


def ckpt_valid(path: Path) -> bool:
    """체크포인트 유효성: config 존재 + weight 파일이 실제로 있고 0바이트 아님 (B2)."""
    path = Path(path)
    has_cfg = (path / "adapter_config.json").exists() or (path / "config.json").exists()
    if not has_cfg:
        return False
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    return any(w.stat().st_size > 0 for w in weights)


def write_done_marker(out_dir: Path, payload: dict):
    """스테이지 완료 마커 — 내용 해시 포함. 크래시 잔해와 완료를 구분한다 (B2)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".done.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_dir / ".done")


def read_done_marker(out_dir: Path):
    p = Path(out_dir) / ".done"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def stage_done(out_dir: Path, expect: dict = None) -> bool:
    """완료 여부: .done 존재 + (expect의 키들이 마커와 일치)."""
    marker = read_done_marker(out_dir)
    if marker is None:
        return False
    for k, v in (expect or {}).items():
        if marker.get(k) != v:
            return False
    return True


def jsonl_rows(path: Path) -> int:
    """jsonl 유효 행수 (부분 작성 산출물 검증용, B2)."""
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                json.loads(line)
                n += 1
            except Exception:
                return -1     # 깨진 행 → 부분 작성으로 간주
    return n
