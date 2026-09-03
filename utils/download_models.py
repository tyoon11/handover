"""
download_models.py — 모델 다운로드 (v3.2)

v3.1 대비 바뀐 점
  - 레지스트리 중복 제거: 대상 목록을 **config_v3.MODELS 한 곳**에서 읽는다.
    (v3.1은 이 파일에 별도 dict가 있어서 repo/dir 이 config와 어긋날 수 있었다)
  - 다운로드 **계획**을 먼저 보여준다: 용량 합계 vs 실제 디스크 여유 → 부족하면 시작 안 함.
    (78~93GB 모델을 반쯤 받다 디스크가 차는 사고 방지)
  - 그룹 프리셋: --group base|teacher|judge|v32|all
  - 받은 뒤 **검증**: config.json + weight 파일 + 실측 용량이 등록값의 90% 이상
  - manifest 기록: 어떤 repo revision 을 받았는지 남긴다 (재현성 — 폐쇄망에서 중요)

repo ID·용량은 2026-09-03 HF 조회로 실측 확인했다. 전량(13개) = 약 540 GB.

실행
  python utils/download_models.py --check                      # 상태만
  python utils/download_models.py --group v32                  # v3.2에 새로 필요한 것만
  python utils/download_models.py --models qwen35_122b prometheus
  python utils/download_models.py --group all --token <HF_TOKEN>
  HANDOVER_MODEL_DIR=/data/local_models python utils/download_models.py --group judge

gated 모델(llama, gemma4, gemma4_31b, medgemma27b)은 HF 웹에서 라이선스 동의 후
--token 또는 HF_TOKEN 환경변수가 있어야 받아진다.

병원 프록시 TLS (CERTIFICATE_VERIFY_FAILED: self-signed certificate in chain)
---------------------------------------------------------------------------
병원망은 TLS 를 가로채기 때문에 huggingface.co 인증서 체인 끝에 병원 자체 CA 가 붙는다.
파이썬 기본 신뢰목록(certifi)에는 그 CA 가 없어 검증이 실패한다. 해결 순서:

  0) 대개 시스템 번들에 이미 병원 CA 가 들어있다. 그걸 지정하면 끝난다.
       python utils/download_models.py --probe --ca-bundle /etc/ssl/certs/ca-certificates.crt
       export HANDOVER_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
     ※ 이 컨테이너는 REQUESTS_CA_BUNDLE 이 이미 시스템 번들을 가리키고 있다. 그런데
        **httpx 는 REQUESTS_CA_BUNDLE·SSL_CERT_FILE 을 읽지 않는다** (certifi 로 컨텍스트를
        직접 만든다). huggingface_hub 가 requests → httpx 로 바뀌면서 "전엔 됐는데 지금
        안 되는" 상황이 생긴 원인이다. 그래서 env 가 아니라 verify 값으로 넘겨야 한다.

  1) 파이썬 전체를 한 번에 고친다 (pip·datasets·vllm 다운로드까지 — 권장)
       python utils/download_models.py --fix-certifi
     certifi 번들에 시스템 번들의 누락 CA 만 덧붙인다(백업 생성, 재실행 안전).

  2) 프록시 CA 를 직접 뽑아 지정한다 (시스템 번들에도 없을 때)
       python utils/download_models.py --extract-ca ~/hf_proxy_ca.pem
       python utils/download_models.py --probe --ca-bundle ~/hf_proxy_ca.pem

  3) 최후수단 — 검증을 끈다 (신뢰된 병원망 안에서만, 토큰이 미검증 TLS 로 나간다)
       python utils/download_models.py --group v32 --insecure

TLS 설정은 huggingface_hub 의 공식 훅 configure_http_backend() 로 주입하고,
그게 없는 구버전에서는 httpx/requests 를 직접 패치한다.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_v3.config_v3 import (      # noqa: E402
    MODELS, MODEL_BASE, EVAL_JUDGE_CANDIDATES, PAIRGEN_JUDGE, TEACHER_CANDIDATES,
    model_path, model_size_gb, model_role,
)

# v3.2에서 새로 필요한 것 (v3.1까지 이미 받아둔 base 6종 제외)
V32_NEW = TEACHER_CANDIDATES + [PAIRGEN_JUDGE] + EVAL_JUDGE_CANDIDATES

IGNORE_PATTERNS = ["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                   "rust_model*", "coreml*", "onnx*", "*.gguf"]

MANIFEST = "download_manifest.json"


def group_keys(group: str) -> list:
    if group == "all":
        return list(MODELS)
    if group == "v32":
        seen, out = set(), []
        for k in V32_NEW:                      # 중복 제거 + 순서 유지
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out
    return [k for k in MODELS if model_role(k) == group]


def check_status(key: str):
    """(완료여부, 설명, 실측GB)"""
    d = model_path(key)
    if not d.exists():
        return False, "폴더 없음", 0.0
    if not (d / "config.json").exists():
        return False, "config.json 없음", 0.0
    weights = list(d.rglob("*.safetensors")) + list(d.rglob("*.bin"))
    if not weights:
        return False, "weight 파일 없음", 0.0
    gb = sum(f.stat().st_size for f in weights) / 1e9
    want = model_size_gb(key)
    if want and gb < want * 0.9:
        return False, f"용량 부족 {gb:.1f}/{want:.1f} GB (중단된 다운로드)", gb
    return True, f"{len(weights)}개 파일 {gb:.1f} GB", gb


def print_plan(targets: list) -> float:
    print(f"\n[저장 경로] {MODEL_BASE}")
    print(f"  {'키':<13} {'역할':<8} {'용량':>8}  {'상태':<34} repo")
    print("  " + "-" * 108)
    need = 0.0
    for k in targets:
        ok, detail, _gb = check_status(k)
        if not ok:
            need += model_size_gb(k)
        lock = "🔒" if MODELS[k].get("gated") else "  "
        print(f"  {'✓' if ok else '·'} {k:<11} {model_role(k):<8} "
              f"{model_size_gb(k):7.1f}G  {detail:<34} {lock}{MODELS[k]['repo']}")
    free = shutil.disk_usage(str(MODEL_BASE if MODEL_BASE.exists()
                                 else MODEL_BASE.parent if MODEL_BASE.parent.exists()
                                 else Path.cwd())).free / 1e9
    print(f"\n  받아야 할 용량 {need:.1f} GB · 디스크 여유 {free:.1f} GB", end="")
    if need and free < need * 1.1:
        print("  ← ⚠ 여유 부족 (10% 마진 포함)")
    else:
        print()
    return need


def download_one(key: str, token, force: bool, revision=None) -> bool:
    from huggingface_hub import snapshot_download

    info = MODELS[key]
    local = model_path(key)

    ok, detail, _ = check_status(key)
    if ok and not force:
        print(f"  [SKIP] {key}: 완료 ({detail})")
        return True
    if info.get("gated") and not token:
        print(f"  [WARN] {key}: gated 모델 — 토큰 없으면 401 이 난다")

    print(f"\n  ▶ {key}  {info['repo']}  ({model_size_gb(key):.1f} GB, {model_role(key)})")
    print(f"    → {local}")
    t0 = time.time()
    try:
        path = snapshot_download(repo_id=info["repo"], local_dir=str(local), token=token,
                                 revision=revision, ignore_patterns=IGNORE_PATTERNS,
                                 max_workers=8)
        ok2, detail2, gb = check_status(key)
        mins = (time.time() - t0) / 60
        if ok2:
            print(f"    ✓ 완료 {detail2} · {mins:.1f}분 "
                  f"({gb / max(mins * 60, 1e-9) * 1000:.0f} MB/s)")
            _write_manifest(key, path, gb)
            return True
        print(f"    ✗ 검증 실패: {detail2}")
        return False
    except Exception as e:
        print(f"    ✗ 오류: {type(e).__name__}: {str(e)[:300]}")
        if "401" in str(e) or "gated" in str(e).lower():
            print("      → HF 웹에서 라이선스 동의 후 --token 재시도")
        if "No space" in str(e) or "ENOSPC" in str(e):
            print("      → 디스크 부족. --check 로 계획 다시 확인")
        return False


def _write_manifest(key: str, path: str, gb: float):
    """어떤 revision 을 받았는지 기록 (폐쇄망 재현성)."""
    mf = MODEL_BASE / MANIFEST
    try:
        data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    except Exception:
        data = {}
    rev = None
    for p in (Path(path) / ".cache" / "huggingface" / "download",):
        if p.exists():
            break
    try:                    # local_dir 모드에서도 남는 커밋 해시가 있으면 사용
        rc = Path(path) / ".cache" / "huggingface" / ".commit_hash"
        rev = rc.read_text().strip() if rc.exists() else None
    except Exception:
        rev = None
    data[key] = dict(repo=MODELS[key]["repo"], dir=MODELS[key]["dir"],
                     size_gb=round(gb, 1), revision=rev,
                     downloaded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"      (manifest 기록 실패: {e})")


def _resolve_tls(ca_bundle: str = None, insecure: bool = False):
    """(verify_value, 설명) — httpx/requests 에 넘길 verify 값을 결정한다.

    우선순위: --insecure / HANDOVER_INSECURE_SSL=1  >  --ca-bundle / HANDOVER_CA_BUNDLE
              >  기존 SSL_CERT_FILE·REQUESTS_CA_BUNDLE  >  기본(certifi)
    """
    if insecure or os.environ.get("HANDOVER_INSECURE_SSL") == "1":
        return False, "TLS 검증 끔 (--insecure)"
    ca = ca_bundle or os.environ.get("HANDOVER_CA_BUNDLE")
    if ca:
        ca = str(Path(ca).expanduser())
        if not Path(ca).exists():
            sys.exit(f"CA 파일 없음: {ca}  (--extract-ca 로 먼저 뽑을 것)")
        return ca, f"CA 지정 {ca}"
    for env in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        v = os.environ.get(env)
        if v and Path(v).exists():
            return v, f"CA (env {env}) {v}"
    return True, "기본 신뢰목록(certifi)"


def _configure_http(verify):
    """huggingface_hub 의 HTTP 백엔드에 verify 를 주입.

    공식 훅(configure_http_backend)이 있으면 그걸 쓰고, 없으면 httpx/requests 를 패치한다.
    httpx 는 기본 컨텍스트를 certifi 로 만들기 때문에 SSL_CERT_FILE env 만으로는
    안 먹는 경우가 있다 → verify 를 값으로 직접 넘기는 이 경로가 확실하다.
    """
    if verify is False:
        print("[WARN] TLS 검증 비활성 — HF 토큰이 미검증 연결로 전송된다 (신뢰망 전용)")
        try:
            import urllib3
            urllib3.disable_warnings()
        except Exception:
            pass
    if verify is True:
        return                      # 기본값 그대로

    # env 도 같이 세팅 (하위 프로세스·requests 경로 대비)
    if isinstance(verify, str):
        for env in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(env, verify)

    injected = False
    try:
        import huggingface_hub as _hf
        from huggingface_hub import configure_http_backend

        # 백엔드가 httpx 인지 requests 인지 **추측하지 말고 확인**한다.
        #   hub >= 0.30 은 httpx, 그 이전은 requests. httpx 가 깔려 있다는 사실만으로
        #   httpx.Client 를 requests 기반 hub 에 주입하면 그대로 깨진다.
        backend = "requests"
        try:
            import huggingface_hub.utils._http as _hh
            if getattr(_hh, "httpx", None) is not None:
                backend = "httpx"
        except Exception:
            ver = getattr(_hf, "__version__", "0")
            try:
                major, minor = (int(x) for x in ver.split(".")[:2])
                backend = "httpx" if (major, minor) >= (0, 30) else "requests"
            except Exception:
                pass

        if backend == "httpx":
            import httpx

            def _httpx_factory(v=verify):
                return httpx.Client(verify=v, follow_redirects=True, timeout=60.0)
            configure_http_backend(backend_factory=_httpx_factory)
        else:
            import requests

            def _req_factory(v=verify):
                s = requests.Session()
                s.verify = v
                return s
            configure_http_backend(backend_factory=_req_factory)
        injected = True
        print(f"  (TLS: huggingface_hub {getattr(_hf, '__version__', '?')} · "
              f"{backend} 백엔드에 주입)")
    except Exception as e:
        print(f"  (configure_http_backend 사용 불가: {type(e).__name__}: "
              f"{str(e)[:80]} — 직접 패치로 전환)")

    if injected:
        return

    # 폴백: 라이브러리 클래스 자체를 패치 (구버전 huggingface_hub)
    try:
        import httpx
        for cls in (httpx.Client, httpx.AsyncClient):
            orig = cls.__init__

            def patched(self, *a, _orig=orig, _v=verify, **kw):
                kw["verify"] = _v
                _orig(self, *a, **kw)
            cls.__init__ = patched
        print("  (TLS: httpx 직접 패치)")
    except ImportError:
        pass
    try:
        import requests
        orig_req = requests.Session.request

        def patched_req(self, *a, _orig=orig_req, _v=verify, **kw):
            kw.setdefault("verify", _v)
            return _orig(self, *a, **kw)
        requests.Session.request = patched_req
        print("  (TLS: requests 직접 패치)")
    except ImportError:
        pass


HF_HOST = "huggingface.co"


def extract_ca(out_path: str, host: str = HF_HOST) -> bool:
    """openssl 로 서버가 제시하는 인증서 체인을 받아 PEM 으로 저장한다.

    병원 프록시가 가로채면 체인 끝에 병원 자체 CA 가 들어있다. 체인 전체를 저장해도
    무해하므로 그대로 CA bundle 로 쓴다. openssl 이 없으면 실패를 알리고 수동 안내.
    """
    import re
    import subprocess
    out = Path(out_path).expanduser()
    cmd = ["openssl", "s_client", "-showcerts", "-servername", host,
           "-connect", f"{host}:443"]
    print(f"[CA 추출] {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, input=b"", capture_output=True, timeout=60)
    except FileNotFoundError:
        print("  ✗ openssl 이 없다. 수동 방법:")
        print("    - 브라우저에서 huggingface.co 인증서 → 발급자(병원 CA) 를 PEM 으로 내보내기")
        print("    - 또는 컨테이너의 기존 번들 사용: /etc/ssl/certs/ca-certificates.crt")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ 연결 타임아웃 — 프록시 설정(HTTPS_PROXY) 확인 필요")
        return False
    text = r.stdout.decode("utf-8", "ignore")
    certs = re.findall(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                       text, re.DOTALL)
    if not certs:
        print("  ✗ 인증서를 받지 못했다. openssl 출력 앞부분:")
        print("   ", (r.stderr.decode("utf-8", "ignore") or text)[:400])
        return False
    issuers = re.findall(r"^\s*\d+\s+s:(.*)$", text, re.M)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(certs) + "\n", encoding="utf-8")
    print(f"  ✓ 인증서 {len(certs)}개 저장 → {out}")
    for i, sub in enumerate(issuers):
        print(f"    [{i}] {sub.strip()[:100]}")
    print(f"\n  다음: python utils/download_models.py --probe --ca-bundle {out}")
    return True


SYSTEM_CA = "/etc/ssl/certs/ca-certificates.crt"


def _pem_blocks(text: str) -> list:
    import re
    return re.findall(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                      text, re.DOTALL)


def fix_certifi(src: str = SYSTEM_CA) -> bool:
    """certifi 번들에 시스템 번들의 '없는 CA만' 덧붙인다 (백업 생성, idempotent).

    왜 필요한가: httpx 는 REQUESTS_CA_BUNDLE / SSL_CERT_FILE 을 무시하고 certifi 를 쓴다.
    certifi 를 고쳐두면 이 환경의 **모든 파이썬 HTTPS**(huggingface_hub·pip·datasets·vllm)가
    병원 프록시를 통과한다. 매 명령에 --ca-bundle 을 붙이지 않아도 된다.
    """
    srcp = Path(src)
    if not srcp.exists():
        print(f"  ✗ 시스템 번들 없음: {src}")
        return False
    try:
        import certifi
    except ImportError:
        print("  ✗ certifi 가 없다 (pip install certifi)")
        return False

    dst = Path(certifi.where())
    have = set(_pem_blocks(dst.read_text(encoding="utf-8", errors="ignore")))
    incoming = _pem_blocks(srcp.read_text(encoding="utf-8", errors="ignore"))
    missing = [b for b in incoming if b not in have]

    print(f"[fix-certifi] certifi {dst}")
    print(f"  기존 {len(have)}개 · 시스템 번들 {len(incoming)}개 · 추가 대상 {len(missing)}개")
    if not missing:
        print("  이미 필요한 CA 가 다 들어있다 (변경 없음)")
        return True

    bak = dst.with_suffix(dst.suffix + ".bak")
    if not bak.exists():
        try:
            bak.write_bytes(dst.read_bytes())
            print(f"  백업 생성 {bak}")
        except Exception as e:
            print(f"  ✗ 백업 실패({e}) — 쓰기 권한 문제면 --ca-bundle 방식을 쓸 것")
            return False
    try:
        with open(dst, "a", encoding="utf-8") as f:
            f.write("\n# --- appended from " + src + " ---\n")
            f.write("\n".join(missing) + "\n")
    except Exception as e:
        print(f"  ✗ 쓰기 실패({e}) — 권한 없으면 export HANDOVER_CA_BUNDLE={src} 로 우회")
        return False
    print(f"  ✓ CA {len(missing)}개 추가. 복구는  cp {bak} {dst}")
    return probe(True)


def probe(verify, host: str = HF_HOST) -> bool:
    """현재 TLS 설정으로 실제 핸드셰이크를 해본다 (huggingface_hub 없이도 동작)."""
    import socket
    import ssl
    print(f"[probe] {host}:443 · {verify if verify is not True else 'certifi 기본'}")
    if verify is False:
        ctx = ssl._create_unverified_context()
    elif isinstance(verify, str):
        ctx = ssl.create_default_context(cafile=verify)
    else:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
    try:
        with socket.create_connection((host, 443), timeout=20) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ss:
                cert = ss.getpeercert() or {}
                iss = dict(x[0] for x in cert.get("issuer", ()) if x)
                sub = dict(x[0] for x in cert.get("subject", ()) if x)
                print(f"  ✓ 핸드셰이크 성공 · TLS {ss.version()}")
                print(f"    subject={sub.get('commonName')} / issuer={iss.get('commonName')}"
                      f" ({iss.get('organizationName')})")
                return True
    except ssl.SSLCertVerificationError as e:
        print(f"  ✗ 인증서 검증 실패: {e.verify_message or e}")
        print("    → --extract-ca 로 프록시 CA 를 뽑아 --ca-bundle 로 지정할 것")
        return False
    except Exception as e:
        print(f"  ✗ 연결 실패: {type(e).__name__}: {str(e)[:200]}")
        print("    → 프록시 env(HTTPS_PROXY/HTTP_PROXY/NO_PROXY) 확인")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="모델 다운로드 (대상 목록은 pipeline_v3/config_v3.py MODELS 가 단일 소스)")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=None)
    ap.add_argument("--group", choices=["base", "teacher", "judge", "v32", "all"],
                    default=None, help="프리셋 (v32 = teacher후보+판정후보)")
    ap.add_argument("--check", action="store_true", help="계획·상태만 출력")
    ap.add_argument("--force", action="store_true", help="완료된 것도 다시 받기")
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--revision", type=str, default=None,
                    help="특정 커밋/태그 고정 (모델 1개 받을 때만 권장)")
    ap.add_argument("--yes", action="store_true", help="확인 프롬프트 없이 진행")
    ap.add_argument("--ca-bundle", dest="ca_bundle", type=str, default=None,
                    help="병원 프록시 CA PEM 경로 (env HANDOVER_CA_BUNDLE 도 가능)")
    ap.add_argument("--insecure", action="store_true",
                    help="TLS 검증 끔 — 신뢰된 병원망 전용 (토큰이 미검증 연결로 나간다)")
    ap.add_argument("--extract-ca", dest="extract_ca", type=str, default=None,
                    help="프록시 CA 체인을 PEM 으로 뽑아 저장하고 종료")
    ap.add_argument("--probe", action="store_true",
                    help="TLS 핸드셰이크만 시험하고 종료")
    ap.add_argument("--fix-certifi", dest="fix_certifi", nargs="?", const=SYSTEM_CA,
                    default=None, metavar="SYSTEM_BUNDLE",
                    help=f"certifi 번들에 시스템 CA 를 덧붙여 파이썬 전체를 고친다 "
                         f"(기본 {SYSTEM_CA})")
    args = ap.parse_args()

    if args.fix_certifi:
        sys.exit(0 if fix_certifi(args.fix_certifi) else 1)
    if args.extract_ca:
        sys.exit(0 if extract_ca(args.extract_ca) else 1)

    verify, how = _resolve_tls(args.ca_bundle, args.insecure)
    print(f"[TLS] {how}")
    if args.probe:
        sys.exit(0 if probe(verify) else 1)
    _configure_http(verify)

    targets = args.models or group_keys(args.group or "v32")
    token = (args.token or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"[인증] {'HF 토큰 사용 ' + token[:6] + '...' if token else '토큰 없음 (public 만)'}")
    print(f"[대상] {len(targets)}개: {', '.join(targets)}")

    MODEL_BASE.mkdir(parents=True, exist_ok=True)
    need = print_plan(targets)
    if args.check:
        return
    if need == 0:
        print("\n받을 게 없다 (전부 완료). --force 로 강제 재다운로드 가능.")
        return
    if not args.yes and sys.stdin.isatty():
        if input(f"\n{need:.1f} GB 다운로드를 시작할까? [y/N] ").strip().lower() != "y":
            print("취소")
            return

    results = {k: download_one(k, token, args.force, args.revision) for k in targets}

    print("\n" + "=" * 60)
    ok = [k for k, v in results.items() if v]
    bad = [k for k, v in results.items() if not v]
    if ok:
        print(f"✓ 성공 {len(ok)}: {', '.join(ok)}")
    if bad:
        print(f"✗ 실패 {len(bad)}: {', '.join(bad)}")
        print("  gated 모델은 HF 라이선스 동의 + --token, 중단된 건 그냥 재실행하면 이어받는다")
        sys.exit(1)
    print(f"  manifest: {MODEL_BASE / MANIFEST}")


if __name__ == "__main__":
    main()
