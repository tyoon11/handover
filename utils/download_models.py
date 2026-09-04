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


def _stored_token_path() -> Path:
    """HF CLI 가 토큰을 저장하는 표준 위치. env 로 옮겨졌으면 그쪽."""
    if os.environ.get("HF_TOKEN_PATH"):
        return Path(os.environ["HF_TOKEN_PATH"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "token"


def resolve_token(cli_token: str = None):
    """(token, 출처설명) — 토큰 값은 절대 출력하지 않는다.

    우선순위: --token > HF_TOKEN/HUGGING_FACE_HUB_TOKEN env > 저장 파일.
    저장 파일이 있으면 huggingface_hub 가 token=None 으로도 알아서 쓰지만,
    '토큰 없음'이라고 잘못 안내하지 않도록 여기서 같이 확인한다.
    """
    if cli_token:
        return cli_token, "--token 인자"
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v, f"env {env}"
    tp = _stored_token_path()
    try:
        if tp.exists():
            v = tp.read_text(encoding="utf-8").strip()
            if v:
                return v, f"저장 파일 {tp}"
    except Exception:
        pass
    return None, ("없음 — public 모델만. 미인증은 rate limit 이 걸려 느리다.\n"
                  f"           토큰 저장: printf '%s' '<hf_토큰>' > {tp} && chmod 600 {tp}")


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
    weights = [f for f in (list(d.rglob("*.safetensors")) + list(d.rglob("*.bin")))
               if not _is_junk(f.relative_to(d))]
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


def download_one(key: str, token, force: bool, revision=None,
                 max_workers: int = 8) -> bool:
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
                                 max_workers=max_workers)
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
    """huggingface_hub 의 HTTP 계층에 verify 를 주입.

    순서가 중요하다: **httpx/requests 를 먼저 패치**하고 그다음 hub 를 건드린다.
    hub 를 먼저 import 하면 그 시점에 만들어진 클라이언트가 옛 verify 를 물고 있어
    나중 패치가 안 먹는다. (구버전 hub 는 configure_http_backend 가 없거나,
    httpx 로 이전한 최신 hub 는 그 함수를 제거했다 — 둘 다 이 경로로 커버된다.)
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

    if isinstance(verify, str):     # 하위 프로세스·requests 경로 대비
        for env in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(env, verify)

    patched = []
    try:                            # ① httpx (최신 hub 가 쓰는 백엔드)
        import httpx
        for cls in (httpx.Client, httpx.AsyncClient):
            orig = cls.__init__

            def _init(self, *a, _orig=orig, _v=verify, **kw):
                kw["verify"] = _v
                _orig(self, *a, **kw)
            cls.__init__ = _init
        patched.append("httpx")
    except ImportError:
        pass
    try:                            # ② requests (구버전 hub)
        import requests
        orig_req = requests.Session.request

        def _request(self, *a, _orig=orig_req, _v=verify, **kw):
            kw.setdefault("verify", _v)
            return _orig(self, *a, **kw)
        requests.Session.request = _request
        patched.append("requests")
    except ImportError:
        pass

    # ③ 공식 훅이 있으면 추가로 등록 (없으면 ①②로 충분하다 — 경고 아님)
    hook = ""
    try:
        import huggingface_hub as _hf
        ver = getattr(_hf, "__version__", "?")
        try:
            from huggingface_hub import configure_http_backend
            backend = "requests"
            try:
                import huggingface_hub.utils._http as _hh
                if getattr(_hh, "httpx", None) is not None:
                    backend = "httpx"
            except Exception:
                pass
            if backend == "httpx":
                import httpx as _hx

                def _factory(v=verify):
                    return _hx.Client(verify=v, follow_redirects=True, timeout=120.0)
            else:
                import requests as _rq

                def _factory(v=verify):
                    ss = _rq.Session()
                    ss.verify = v
                    return ss
            configure_http_backend(backend_factory=_factory)
            hook = f" + configure_http_backend({backend})"
        except ImportError:
            hook = " (configure_http_backend 없음 — 패치로 충분)"
        print(f"  (TLS 주입: {'/'.join(patched) or '없음'}{hook} · hub {ver})")
    except Exception as e:
        print(f"  (TLS 주입: {'/'.join(patched) or '없음'} · hub 확인 실패 {type(e).__name__}: "
              f"{str(e)[:60]})")


def _tune_hub_env(disable_xet=None, timeout: int = 120, workers_note: str = ""):
    """프록시 환경에서 **대용량 샤드만 0바이트로 멈추는** 문제 대응.

    관측된 증상: config.json 등 작은 파일은 받아지는데 *.safetensors 는 .incomplete 가
    0바이트로 고정. 원인은 파일 종류에 따라 **호스트가 달라지는 것**이다.
      - 작은 파일 : huggingface.co 에서 그대로 내려온다 → 프록시 통과
      - 대용량    : Xet 스토리지(cas-bridge/transfer.xethub.hf.co)로 리다이렉트되고
                    자체 청크 프로토콜을 쓴다 → 병원/기업 프록시에서 자주 막힌다
    그래서 프록시가 감지되면 Xet 을 끄고 기존 LFS CDN 경로로 받게 한다.
    HF_HUB_DOWNLOAD_TIMEOUT 기본 10초도 프록시 경유엔 너무 짧아 조용한 재시도만 반복된다.

    ⚠ 이 env 들은 huggingface_hub **import 전에** 설정해야 반영된다.
    """
    proxy = [k for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
             if os.environ.get(k)]
    if disable_xet is None:
        disable_xet = bool(proxy)
    if proxy:
        print(f"  (프록시 감지: {', '.join(proxy)})")
    if disable_xet and "HF_HUB_DISABLE_XET" not in os.environ:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    xet_off = os.environ.get("HF_HUB_DISABLE_XET", "0") == "1"
    try:
        import hf_xet          # noqa: F401
        has_xet = True
    except Exception:
        has_xet = False
    print(f"  (Xet: {'비활성' if xet_off else '활성'}"
          f" · hf_xet {'설치됨' if has_xet else '없음'}"
          f"{' — 대용량은 LFS CDN(cdn-lfs-*.hf.co) 경로로 받는다' if xet_off else ''})")
    if "HF_HUB_DOWNLOAD_TIMEOUT" not in os.environ:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
        print(f"  (HF_HUB_DOWNLOAD_TIMEOUT={timeout})")
    if workers_note:
        print(f"  ({workers_note})")


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
    _px = _proxy_url()
    if _px:                     # 직결이 없는 컨테이너 → openssl 도 프록시를 타야 한다
        _ph, _pp = _proxy_hostport(_px)
        cmd += ["-proxy", f"{_ph}:{_pp}"]
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


class _Redirected(Exception):
    def __init__(self, url):
        self.url = url


def _ssl_ctx(verify):
    import ssl
    if verify is False:
        return ssl._create_unverified_context()
    if isinstance(verify, str):
        return ssl.create_default_context(cafile=verify)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


PROXY_PROBE_HOSTS = [
    ("huggingface.co", "HF API·소용량 (허용 확인됨)"),
    ("us.aws.cdn.hf.co", "대용량 리다이렉트 목적지 (문제 호스트)"),
    ("cdn-lfs-us-1.hf.co", "클래식 LFS CDN"),
    ("cas-bridge.xethub.hf.co", "Xet CAS"),
    ("example.com", "무관한 외부 — 기본정책 판별(allowlist vs blocklist)"),
    ("pypi.org", "pip 미러가 아닌 원본"),
]


def probe_proxy(hosts=None, timeout: int = 25) -> bool:
    """프록시가 호스트별로 어떻게 응답하는지 CONNECT 만 보내 확인한다.

    왜: '막혔다'의 원인이 두 갈래인데 대응이 다르다.
      - squid ACL 거부       → 즉시 **403**(+ X-Squid-Error) / 인증필요면 407
      - squid 는 허용, 상단 방화벽이 drop → **무응답(타임아웃)** 또는 503/504
    TLS 이전 단계라 인증서·CA 와 무관하게 경로만 본다.

    example.com 결과가 특히 중요하다 — 이것이 통과하면 프록시는 allowlist 방식이 아니고,
    그러면 '허용 목록에 추가' 요청이 아니라 '해당 대역 차단 해제' 요청이어야 한다.
    """
    import socket
    import time as _t

    proxy = _proxy_url()
    if not proxy:
        print("프록시 env 가 없다 (HTTPS_PROXY/https_proxy) — 이 진단은 프록시 환경 전용")
        return False
    phost, pport = _proxy_hostport(proxy)
    print(f"[probe-proxy] {phost}:{pport} · CONNECT 만 보내 응답을 본다 "
          f"(타임아웃 {timeout}s)\n")
    print(f"  {'호스트':<26} {'경과':>6}  {'응답':<34} 판정")
    print("  " + "-" * 96)

    verdicts = {}
    for host, note in (hosts or PROXY_PROBE_HOSTS):
        t0 = _t.time()
        status, extra, verdict = "", "", ""
        try:
            sock = socket.create_connection((phost, pport), timeout=timeout)
            sock.settimeout(timeout)
            sock.sendall((f"CONNECT {host}:443 HTTP/1.1\r\n"
                          f"Host: {host}:443\r\n"
                          f"User-Agent: handover-proxy-probe\r\n\r\n").encode())
            resp = b""
            while b"\r\n\r\n" not in resp and len(resp) < 65536:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
            sock.close()
            head = resp.decode("utf-8", "ignore")
            status = head.split("\r\n", 1)[0].strip() or "(빈 응답)"
            for line in head.split("\r\n"):
                if line.lower().startswith(("x-squid-error", "x-cache", "server:")):
                    extra += (" | " if extra else "") + line.strip()
            code = ""
            for tok in status.split():
                if tok.isdigit():
                    code = tok
                    break
            if code == "200":
                verdict = "허용 + 연결성공"
            elif code == "403":
                verdict = "★ 프록시 ACL 거부 (허용목록에 없음)"
            elif code == "407":
                verdict = "프록시 인증 필요"
            elif code in ("503", "504"):
                verdict = "★ 프록시는 허용, 상단 연결/DNS 실패"
            else:
                verdict = f"기타(code={code or '?'})"
        except socket.timeout:
            status = "(무응답)"
            verdict = "★ 타임아웃 — 상단 방화벽 drop 가능성"
        except Exception as e:
            status = f"{type(e).__name__}"
            verdict = f"연결 실패: {str(e)[:40]}"
        el = _t.time() - t0
        verdicts[host] = verdict
        print(f"  {host:<26} {el:5.1f}s  {status:<34} {verdict}")
        if extra:
            print(f"  {'':<26} {'':>6}  {extra[:90]}")
        print(f"  {'':<26} {'':>6}  ({note})")

    print("\n  읽는 법")
    print("   · 403 이면 squid 허용목록 문제 → '*.hf.co 추가' 요청이 맞다")
    print("   · 무응답/503 이면 squid 는 통과시켰고 그 위(방화벽·egress)에서 막힌 것")
    print("     → 요청 대상이 프록시 담당자가 아니라 네트워크 보안 담당일 수 있다")
    ex = verdicts.get("example.com", "")
    if ex.startswith("허용"):
        print("   · example.com 이 통과했다 → 프록시는 **allowlist 방식이 아니다**.")
        print("     그러면 'us.aws.cdn.hf.co 만' 막힌 것이므로 IP 대역·카테고리 차단"
              "(CDN/파일공유 분류)일 가능성이 크다.")
    elif ex:
        print("   · example.com 도 막혔다 → 기본거부(allowlist) 방식 확정."
              " '*.hf.co 허용 추가' 요청이 정확하다.")
    return True


def probe_cdn(verify, repo: str = None, timeout: int = 20) -> bool:
    """대용량 파일이 실제로 어느 호스트로 리다이렉트되고, 그 호스트에 닿는지 시험한다.

    왜 필요한가: huggingface.co(API·작은 파일)는 열려 있는데 **대용량만** 막히는 일이
    흔하다. 대용량은 다른 도메인으로 리다이렉트되기 때문이다:
        구(舊)  cdn-lfs*.huggingface.co        ← *.huggingface.co 허용 규칙에 걸림
        현(現)  us.aws.cdn.hf.co / xethub.hf.co ← **hf.co 는 다른 도메인**이라 별도 허용 필요
    프록시 allowlist 가 *.huggingface.co 만 갖고 있으면 예전엔 되던 다운로드가
    조용히(타임아웃까지 0바이트로) 멈춘다. 이 함수가 그 경계를 짚어준다.
    """
    import json as _json
    import urllib.error
    import urllib.request

    repo = repo or MODELS[TEACHER_CANDIDATES[0]]["repo"]
    ctx = _ssl_ctx(verify)

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise _Redirected(newurl)

    opener_nr = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ctx), _NoRedirect)
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))

    print(f"[probe-cdn] repo={repo}")
    # ① API (작은 요청) — 여기가 막히면 망 자체 문제
    try:
        with opener.open(f"https://{HF_HOST}/api/models/{repo}", timeout=timeout) as r:
            meta = _json.loads(r.read().decode())
        files = [s["rfilename"] for s in meta.get("siblings", [])
                 if s["rfilename"].endswith(".safetensors")]
        print(f"  ① API 도달 OK · safetensors {len(files)}개")
    except Exception as e:
        print(f"  ① API 실패: {type(e).__name__}: {str(e)[:160]}")
        return False
    if not files:
        print("  (safetensors 없음 — 다른 repo 로 시험할 것)")
        return False

    # ② 리다이렉트 목적지 확인 (따라가지 않는다)
    url = f"https://{HF_HOST}/{repo}/resolve/main/{files[0]}"
    target = None
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
        with opener_nr.open(req, timeout=timeout) as r:
            print(f"  ② 리다이렉트 없음 (HTTP {r.status}) — huggingface.co 가 직접 서빙")
            return True
    except _Redirected as e:
        target = e.url
    except Exception as e:
        print(f"  ② 실패: {type(e).__name__}: {str(e)[:160]}")
        return False

    from urllib.parse import urlparse
    cdn_host = urlparse(target).netloc
    print(f"  ② 대용량 리다이렉트 → {cdn_host}")

    # ③ 그 호스트에서 1KB 만 받아본다 — 여기서 걸리면 allowlist 문제
    import time as _t
    t0 = _t.time()
    try:
        req = urllib.request.Request(target, headers={"Range": "bytes=0-1023"})
        with opener.open(req, timeout=timeout) as r:
            n = len(r.read())
        print(f"  ③ CDN 도달 OK · {n} bytes / {_t.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"  ③ CDN 실패({_t.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:160]}")
        if "xet" in cdn_host or "xet-bridge" in target:
            print("\n  → 이 목적지는 **Xet** 경로다. Xet 을 끄면 다른 CDN 호스트"
                  "(cdn-lfs-us-1.hf.co 등)로 리다이렉트되므로 그쪽이 열려 있으면 통과한다:")
            print("       HF_HUB_DISABLE_XET=1 python utils/download_models.py --models <키>")
            print("     (pip uninstall hf_xet 로 아예 제거해도 같은 효과)")
        print(f"\n  ⚠ huggingface.co 는 열려 있는데 {cdn_host} 가 막혀 있다.")
        print("     프록시 allowlist 에 아래 도메인 추가를 요청해야 한다:")
        print("       *.hf.co  (us.aws.cdn.hf.co · cas-bridge.xethub.hf.co ·")
        print("                 transfer.xethub.hf.co · cdn-lfs*.hf.co)")
        print("     타임아웃을 키우거나 워커를 줄여도 해결되지 않는다.")
        return False


CHECKSUM_FILE = "checksums.sha256"


# macOS 가 exFAT/FAT 에 만드는 사이드카·메타 파일. 모델 내용이 아니고 전송 중 사라지거나
# 새로 생기므로 해시 대상에서 제외한다 (안 하면 verify 가 대량 오탐을 낸다).
MAC_JUNK_PREFIXES = ("._",)
MAC_JUNK_NAMES = {".DS_Store", ".Spotlight-V100", ".fseventsd", ".Trashes",
                  ".TemporaryItems", ".apdisk"}


def _is_junk(rel: Path) -> bool:
    for part in rel.parts:
        if part.startswith(MAC_JUNK_PREFIXES) or part in MAC_JUNK_NAMES:
            return True
    return False


def _iter_model_files(key: str):
    """모델 디렉토리의 실제 파일들 (HF 내부 캐시·체크섬 파일·macOS 사이드카 제외)."""
    d = model_path(key)
    for f in sorted(d.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(d)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        if rel.name == CHECKSUM_FILE or _is_junk(rel):
            continue
        yield f, rel


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checksums(keys, mode: str) -> bool:
    """물리 이동(외장SSD 등) 전후 무결성 검증.

    write : 받은 쪽에서 모델별 checksums.sha256 생성 (sha256sum 호환 포맷)
    verify: 옮긴 쪽에서 대조. 누락·크기불일치·해시불일치를 각각 구분해 보고한다.

    왜 필요한가: 수십 GB 샤드가 잘려 들어와도 로드는 되고 추론만 이상해지는 경우가 있다.
    크기 비교(--check)로는 못 잡는 손상이 있어 이동 경로에는 해시가 필요하다.
    """
    import time as _t
    ok_all = True
    for key in keys:
        d = model_path(key)
        if not d.exists():
            print(f"  [{key}] 폴더 없음 — 건너뜀")
            ok_all = False
            continue
        cf = d / CHECKSUM_FILE
        files = list(_iter_model_files(key))
        total = sum(f.stat().st_size for f, _ in files)
        print(f"\n  ▶ {key}  {len(files)}개 파일 {total / 1e9:.1f} GB  ({mode})")
        t0 = _t.time()

        if mode == "write":
            lines = []
            for i, (f, rel) in enumerate(files, 1):
                lines.append(f"{_sha256(f)}  {rel.as_posix()}")
                print(f"\r    [{i}/{len(files)}] {rel.as_posix()[:60]:<60}", end="", flush=True)
            cf.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"\r    ✓ {cf} ({len(lines)}개, {(_t.time() - t0) / 60:.1f}분)"
                  f"{' ' * 30}")
            continue

        if not cf.exists():
            print(f"    ✗ {CHECKSUM_FILE} 없음 — 받은 쪽에서 --checksums write 먼저 실행")
            ok_all = False
            continue
        want = {}
        for line in cf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, _, rel = line.partition("  ")
            want[rel.strip()] = h.strip()
        have = {rel.as_posix(): f for f, rel in files}
        missing = [r for r in want if r not in have]
        extra = [r for r in have if r not in want]
        bad = []
        for i, (rel, h) in enumerate(sorted(want.items()), 1):
            if rel in missing:
                continue
            print(f"\r    [{i}/{len(want)}] {rel[:60]:<60}", end="", flush=True)
            if _sha256(have[rel]) != h:
                bad.append(rel)
        print(f"\r    {'✓' if not (missing or bad) else '✗'} "
              f"검증 {len(want) - len(missing) - len(bad)}/{len(want)} 일치 "
              f"({(_t.time() - t0) / 60:.1f}분){' ' * 30}")
        for r in missing:
            print(f"      · 누락 {r}")
        for r in bad:
            print(f"      · 해시 불일치 {r}  ← 재전송 필요")
        for r in extra:
            print(f"      · (목록에 없는 파일 {r})")
        ok_all = ok_all and not (missing or bad)
    return ok_all


def _proxy_url() -> str:
    """HTTPS 용 프록시 URL (없으면 빈 문자열). 폐쇄망 컨테이너는 직결이 없다."""
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(k)
        if v:
            return v
    return ""


def _proxy_hostport(url: str):
    from urllib.parse import urlparse
    u = urlparse(url if "//" in url else "http://" + url)
    return u.hostname, (u.port or 3128)


def probe(verify, host: str = HF_HOST, timeout: int = 20) -> bool:
    """TLS 핸드셰이크 시험 — **프록시가 있으면 CONNECT 터널을 통해** 수행한다.

    직접 socket 으로 붙으면 이 컨테이너에선 [Errno 101] Network is unreachable 이 난다
    (직결 인터넷 없음, squid 만 허용). 터널 안에서 핸드셰이크해야 실제 경로와 같은
    조건이 되고, 프록시가 TLS 를 가로채면 그 인증서 체인이 그대로 보인다.
    """
    import socket
    import ssl

    proxy = _proxy_url()
    ctx = _ssl_ctx(verify)
    route = f"프록시 {proxy} 경유" if proxy else "직결"
    print(f"[probe] {host}:443 · {verify if verify is not True else 'certifi 기본'} · {route}")
    try:
        if proxy:
            phost, pport = _proxy_hostport(proxy)
            sock = socket.create_connection((phost, pport), timeout=timeout)
            req = (f"CONNECT {host}:443 HTTP/1.1\r\n"
                   f"Host: {host}:443\r\n"
                   f"User-Agent: handover-download-probe\r\n\r\n")
            sock.sendall(req.encode())
            resp = b""
            while b"\r\n\r\n" not in resp:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
            status = resp.split(b"\r\n", 1)[0].decode("utf-8", "ignore")
            if " 200" not in status:
                print(f"  ✗ 프록시 CONNECT 거부: {status}")
                print(f"    → {host} 가 프록시 allowlist 에 없다 (관리자 요청 필요)")
                sock.close()
                return False
            print(f"  · CONNECT OK ({status})")
        else:
            sock = socket.create_connection((host, 443), timeout=timeout)

        with ctx.wrap_socket(sock, server_hostname=host) as ss:
            cert = ss.getpeercert() or {}
            iss = dict(x[0] for x in cert.get("issuer", ()) if x)
            sub = dict(x[0] for x in cert.get("subject", ()) if x)
            print(f"  ✓ 핸드셰이크 성공 · TLS {ss.version()}")
            print(f"    subject={sub.get('commonName')} / issuer={iss.get('commonName')}"
                  f" ({iss.get('organizationName')})")
            return True
    except ssl.SSLCertVerificationError as e:
        print(f"  ✗ 인증서 검증 실패: {getattr(e, 'verify_message', None) or e}")
        print("    → --fix-certifi (권장) 또는 --ca-bundle 로 병원 CA 를 지정할 것")
        return False
    except Exception as e:
        print(f"  ✗ 연결 실패: {type(e).__name__}: {str(e)[:200]}")
        if not proxy:
            print("    → 이 환경은 직결 인터넷이 없다. HTTPS_PROXY/https_proxy 를 확인할 것")
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
    ap.add_argument("--max-workers", dest="max_workers", type=int, default=8,
                    help="동시 다운로드 수 (기본 8). 프록시가 동시연결을 조이면 2 로 낮출 것")
    ap.add_argument("--no-xet", dest="no_xet", action="store_true",
                    help="Xet 스토리지 비활성 (프록시 감지 시 기본 동작)")
    ap.add_argument("--xet", dest="xet", action="store_true",
                    help="Xet 강제 사용 (프록시 없는 환경에서 더 빠르다)")
    ap.add_argument("--download-timeout", dest="dl_timeout", type=int, default=120,
                    help="HF_HUB_DOWNLOAD_TIMEOUT 초 (기본 120. HF 기본 10은 프록시에 짧다. "
                         "다만 호스트가 아예 막힌 경우엔 값을 키워도 대기만 길어진다 — "
                         "--probe-cdn 으로 도달성을 먼저 확인할 것)")
    ap.add_argument("--probe-cdn", dest="probe_cdn", nargs="?", const=None, default=False,
                    help="대용량 파일 리다이렉트 체인과 CDN 도달성을 시험 (repo 지정 가능)")
    ap.add_argument("--checksums", choices=["write", "verify"], default=None,
                    help="물리 이동 무결성: write(받은 쪽에서 생성) / verify(옮긴 쪽에서 대조)")
    ap.add_argument("--probe-proxy", dest="probe_proxy", nargs="*", default=None,
                    help="프록시 허용 정책 진단: 호스트별 CONNECT 응답(403=ACL거부 / "
                         "무응답=상단 방화벽). 호스트 나열 가능")
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
    if args.probe_proxy is not None:
        hosts = [(h, "직접 지정") for h in args.probe_proxy] if args.probe_proxy else None
        sys.exit(0 if probe_proxy(hosts) else 1)
    if args.probe_cdn is not False:
        ok_tls = probe(verify)
        sys.exit(0 if (ok_tls and probe_cdn(verify, args.probe_cdn)) else 1)
    # huggingface_hub import 전에 env 를 확정해야 한다 (상수가 import 시점에 읽힌다)
    _tune_hub_env(disable_xet=(False if args.xet else (True if args.no_xet else None)),
                  timeout=args.dl_timeout,
                  workers_note=f"동시 다운로드 {args.max_workers}개")
    _configure_http(verify)

    targets = args.models or group_keys(args.group or "v32")
    token, src = resolve_token(args.token)
    print(f"[인증] {'토큰 사용 (' + src + ')' if token else '토큰 ' + src}")
    print(f"[대상] {len(targets)}개: {', '.join(targets)}")

    MODEL_BASE.mkdir(parents=True, exist_ok=True)
    if args.checksums:
        sys.exit(0 if checksums(targets, args.checksums) else 1)
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

    results = {k: download_one(k, token, args.force, args.revision,
                               max_workers=args.max_workers) for k in targets}

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
