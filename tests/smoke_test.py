"""
tests/smoke_test.py — 단계별 파이프라인 빠른 동작 확인

각 단계를 소량 샘플로 돌려 "돌아가는지" 확인합니다.
--stage data 는 GPU 없이도 동작합니다.

실행 예시:
  python tests/smoke_test.py --stage data
  python tests/smoke_test.py --stage sft    --gpus 4   --base llama
  python tests/smoke_test.py --stage rlaif  --gpus 4,5 --base llama
  python tests/smoke_test.py --stage infer  --gpus 4   --base llama
  python tests/smoke_test.py --stage eval   --gpus 4
  python tests/smoke_test.py --all          --gpus 4,5,6,7 --base llama
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── GPU 설정 (import 전에) ────────────────────────────────────────────────
def _early_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_known_args()[0].gpus


_gpus = _early_parse()
if _gpus:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={_gpus}")

import os, pickle, traceback
import pandas as pd

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def ok(label, fn):
    try:
        r = fn()
        print(f"  {PASS} {label}")
        return r
    except Exception as e:
        print(f"  {FAIL} {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def header(title: str):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


# ── Stage 0: 데이터 & 설정 확인 (GPU 불필요) ──────────────────────────────
def stage_data():
    header("Stage: data — 데이터 & 설정 확인")
    from config import (
        GOLD_PKL, GOLD_REF_PKL, SFT_PKL, SYNTH_PKL,
        VITAL_MAP_PKL, SFT_MODELS, RLAIF_CONFIG, SFT_CONFIG,
    )

    # 파일 존재 확인
    print("\n[1] pkl 파일 존재 여부")
    for name, path in [
        ("GOLD_PKL", GOLD_PKL),
        ("GOLD_REF_PKL", GOLD_REF_PKL),
        ("SFT_PKL", SFT_PKL),
        ("SYNTH_PKL", SYNTH_PKL),
        ("VITAL_MAP_PKL", VITAL_MAP_PKL),
    ]:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            print(f"  {PASS} {name} ({size_mb:.0f}MB)")
        else:
            print(f"  {FAIL} {name} 없음: {path}")

    # 데이터 로드 & 구조 확인
    print("\n[2] 데이터 로드 & 구조")
    synth = ok("SYNTH_PKL 로드", lambda: pd.read_pickle(SYNTH_PKL))
    if synth is not None:
        print(f"       rows={len(synth)}  cols={len(synth.columns)}")
        has_chosen = ok("'chosen' 컬럼 존재", lambda: synth["chosen"].notna().sum())
        has_rejected = ok("'rejected' 컬럼 존재", lambda: synth["rejected"].notna().sum())
        if has_chosen and has_rejected:
            print(f"       chosen 비율: {has_chosen}/{len(synth)}  rejected: {has_rejected}/{len(synth)}")

    vmap = ok("VITAL_MAP_PKL 로드", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if vmap:
        print(f"       vital entries: {len(vmap)}")

    # 모델 경로 확인
    print("\n[3] 모델 경로 확인")
    for name, path in SFT_MODELS.items():
        cfg_path = Path(path) / "config.json"
        if cfg_path.exists():
            print(f"  {PASS} {name}: {path}")
        else:
            print(f"  {WARN} {name}: config.json 없음 ({path})")

    # 설정값 출력
    print("\n[4] RLAIF_CONFIG")
    for k, v in RLAIF_CONFIG.items():
        print(f"       {k}: {v}")

    print(f"\n{PASS} stage_data 완료")


# ── Stage 1: SFT 3 steps ──────────────────────────────────────────────────
def stage_sft(base: str, steps: int = 3, n_samples: int = 8):
    """02_sft_train.py의 train() 함수를 직접 호출해서 실제 코드 경로를 검증."""
    header(f"Stage: sft — {base} · {steps} steps · {n_samples} samples")
    import types, importlib.util, traceback, pandas as pd

    try:
        spec = importlib.util.spec_from_file_location(
            "sft_train", Path(__file__).resolve().parent.parent / "pipeline" / "02_sft_train.py"
        )
        sft_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sft_mod)
        print(f"  {PASS} 02_sft_train 로드")
    except Exception as e:
        print(f"  {FAIL} 02_sft_train 로드: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    args = types.SimpleNamespace(base=base, epochs=1, gpus=None, max_steps=steps)

    # smoke test는 매 스텝마다 log() 경로까지 검증 (시그니처 호환성 사전 검증)
    # save_strategy="no"로 epoch 중간 저장은 비활성화 (final save는 명시 호출이라 영향 없음)
    import config as _cfg
    _orig_sft_cfg = dict(_cfg.SFT_CONFIG)
    _cfg.SFT_CONFIG["logging_steps"] = 1
    _cfg.SFT_CONFIG["save_strategy"] = "no"

    _orig = pd.read_pickle
    pd.read_pickle = lambda p: _orig(p).iloc[:n_samples]
    try:
        sft_mod.train(args)
        print(f"  {PASS} {steps} steps 학습 (실제 Trainer+JudgeAugmentedCollator)")

        # save 검증: final/ 에 추론용 파일이 실제로 생성됐는지 확인
        final_dir = _cfg.SFT_OUT / f"{base}_1ep" / "final"
        has_adapter = (final_dir / "adapter_config.json").exists()
        has_tokenizer = (final_dir / "tokenizer_config.json").exists()
        if not has_adapter:
            print(f"  {FAIL} adapter_config.json 누락: {final_dir}")
            return False
        if not has_tokenizer:
            print(f"  {FAIL} tokenizer_config.json 누락: {final_dir}")
            return False
        print(f"  {PASS} 저장 검증: adapter+tokenizer 파일 모두 존재")
    except Exception as e:
        print(f"  {FAIL} {steps} steps 학습: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        pd.read_pickle = _orig
        _cfg.SFT_CONFIG.clear()
        _cfg.SFT_CONFIG.update(_orig_sft_cfg)

    print(f"\n{PASS} stage_sft 완료 → 02_sft_train.py 실행 가능")
    return True


# ── Stage 2: RLAIF (DPO/SimPO) 3 steps ──────────────────────────────────
def stage_rlaif(base: str, loss: str = "dpo", steps: int = 3, n_samples: int = 8):
    """03_rlaif_train.py의 train() 함수를 직접 호출해서 실제 코드 경로를 검증."""
    header(f"Stage: rlaif — {base} · {loss} · {steps} steps")
    import types, importlib.util, traceback, pandas as pd

    try:
        spec = importlib.util.spec_from_file_location(
            "rlaif_train", Path(__file__).resolve().parent.parent / "pipeline" / "03_rlaif_train.py"
        )
        rlaif_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rlaif_mod)
        print(f"  {PASS} 03_rlaif_train 로드")
    except Exception as e:
        print(f"  {FAIL} 03_rlaif_train 로드: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    args = types.SimpleNamespace(base=base, loss=loss, sft_ckpt=None, gpus=None, max_steps=steps)

    # smoke test는 매 스텝마다 log() 경로까지 검증 (시그니처 호환성 사전 검증)
    import config as _cfg
    _orig_rlaif_cfg = dict(_cfg.RLAIF_CONFIG)
    _cfg.RLAIF_CONFIG["logging_steps"] = 1
    _cfg.RLAIF_CONFIG["save_strategy"] = "no"

    _orig = pd.read_pickle
    pd.read_pickle = lambda p: _orig(p).iloc[:n_samples]
    try:
        rlaif_mod.train(args)
        print(f"  {PASS} {steps} steps 학습 (실제 {loss.upper()}Trainer)")

        # save 검증
        final_dir = _cfg.RLAIF_OUT / f"{base}_raw_{loss}" / "final"
        has_adapter = (final_dir / "adapter_config.json").exists()
        if not has_adapter:
            print(f"  {FAIL} adapter_config.json 누락: {final_dir}")
            return False
        print(f"  {PASS} 저장 검증: adapter 파일 존재")
    except Exception as e:
        print(f"  {FAIL} {steps} steps 학습: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        pd.read_pickle = _orig
        _cfg.RLAIF_CONFIG.clear()
        _cfg.RLAIF_CONFIG.update(_orig_rlaif_cfg)

    print(f"\n{PASS} stage_rlaif 완료 → 03_rlaif_train.py 실행 가능 ({loss})")
    return True


# ── Stage 3: Inference 샘플 2개 ──────────────────────────────────────────
def stage_infer(base: str, n: int = 2):
    header(f"Stage: infer — {base} · {n} samples")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import (
        GOLD_PKL, VITAL_MAP_PKL, SFT_MODELS,
        SYSTEM_PROMPT, build_user_prompt, build_emr_text,
    )

    model_id = str(SFT_MODELS[base])
    print(f"\n  모델: {model_id}")

    tokenizer = ok("토크나이저", lambda: AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    model = ok("모델 로드", lambda: AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ))
    if tokenizer is None or model is None:
        return False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    gold_df = ok("GOLD_PKL", lambda: pd.read_pickle(GOLD_PKL))
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if gold_df is None or vmap is None:
        return False

    print(f"\n  Inference {n}건")
    generated_ok = 0
    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        user = build_user_prompt(emr, vital)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        is_ok = len(gen) >= 2
        print(f"  {'✓' if is_ok else '✗'} [{i+1}] sid={sid}  출력: {gen[:80]!r}")
        if is_ok:
            generated_ok += 1

    if generated_ok == n:
        print(f"\n{PASS} stage_infer 완료 → 04_inference.py 실행 가능")
        return True
    else:
        print(f"\n{FAIL} 출력 실패 {n - generated_ok}/{n}건")
        return False


# ── Stage 4: Evaluate (Judge) 샘플 2개 ───────────────────────────────────
def stage_eval(n: int = 2):
    header(f"Stage: eval — judge {n} samples")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import GOLD_PKL, VITAL_MAP_PKL, EVAL_JUDGE_MODEL, build_user_prompt, build_emr_text
    from pipeline.evaluate_helpers import judge_score, BREVITY_RUBRIC

    tokenizer = ok("Judge 토크나이저", lambda: AutoTokenizer.from_pretrained(EVAL_JUDGE_MODEL, trust_remote_code=True))
    model = ok("Judge 모델 로드", lambda: AutoModelForCausalLM.from_pretrained(
        EVAL_JUDGE_MODEL, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True,
    ))
    if tokenizer is None or model is None:
        return False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    gold_df = ok("GOLD_PKL", lambda: pd.read_pickle(GOLD_PKL))
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if gold_df is None or vmap is None:
        return False

    DUMMY_RESP = "SpO2 93% → O2 보충 유지. EBL 800mL → pRBC 2U 수혈."
    print(f"\n  Scoring {n}건 (dummy response: '{DUMMY_RESP}')")
    scored = 0
    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        instruction = build_user_prompt(emr, vital)
        score = ok(f"[{i+1}] judge_score", lambda: judge_score(model, tokenizer, instruction, DUMMY_RESP, BREVITY_RUBRIC))
        if score is not None:
            print(f"       score: {score}")
            scored += 1

    if scored == n:
        print(f"\n{PASS} stage_eval 완료 → 05_evaluate.py 실행 가능")
        return True
    return False


# ── helpers module 없이 eval 돌릴 수 있도록 인라인 가져오기 ──────────────
def _import_judge():
    """05_evaluate.py에서 judge 함수 직접 임포트."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate", Path(__file__).resolve().parent.parent / "pipeline" / "05_evaluate.py"
    )
    mod = importlib.util.load_from_spec = spec
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def stage_eval_inline(n: int = 2):
    """05_evaluate.py의 judge_score를 직접 import해서 테스트."""
    header(f"Stage: eval — judge {n} samples")
    import torch
    import gc
    from config import GOLD_PKL, VITAL_MAP_PKL, EVAL_JUDGE_MODEL, build_user_prompt, build_emr_text
    import importlib.util

    # 이전 모델 잔류 메모리 해제 (Prometheus/Mixtral MoE 로딩 전 필수)
    gc.collect()
    torch.cuda.empty_cache()
    n_gpu = torch.cuda.device_count()
    free = [torch.cuda.mem_get_info(i)[0] / 1e9 for i in range(n_gpu)]
    print(f"  GPU 여유 메모리: {[f'{f:.1f}GB' for f in free]}")

    spec = importlib.util.spec_from_file_location(
        "ev05",
        Path(__file__).resolve().parent.parent / "pipeline" / "05_evaluate.py",
    )
    ev05 = importlib.util.module_from_spec(spec)
    ok("05_evaluate.py import", lambda: spec.loader.exec_module(ev05))

    # Prometheus(Mixtral MoE) 로딩: max_memory로 GPU별 상한 설정
    _max_mem = {i: "40GiB" for i in range(n_gpu)}
    _orig_load = ev05.load_judge_model

    def _load_with_mem(model_id):
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto",
            max_memory=_max_mem, low_cpu_mem_usage=True, trust_remote_code=True,
        )
        mdl.eval()
        return mdl, tok

    model_and_tok = ok("Judge 모델", lambda: _load_with_mem(EVAL_JUDGE_MODEL))
    if model_and_tok is None:
        return False
    judge_model, judge_tok = model_and_tok

    gold_df = ok("GOLD_PKL", lambda: pd.read_pickle(GOLD_PKL))
    vmap = ok("VITAL_MAP_PKL", lambda: pickle.load(open(VITAL_MAP_PKL, "rb")))
    if gold_df is None or vmap is None:
        return False

    DUMMY = "SpO2 93% → O2 보충. EBL 800mL → pRBC 2U."
    scored = 0
    for i, (_, row) in enumerate(gold_df.iloc[:n].iterrows()):
        emr = build_emr_text(row)
        try:
            v = row["수술 ID"]
            sid = int(v.iloc[0]) if hasattr(v, "iloc") else int(v)
        except Exception:
            sid = -1
        vital = vmap.get(sid, "")
        instr = build_user_prompt(emr, vital)
        score = ok(f"[{i+1}] judge_score", lambda instr=instr: ev05.judge_score(judge_model, judge_tok, instr, DUMMY, ev05.BREVITY_RUBRIC))
        if score is not None:
            print(f"       score: {score}")
            scored += 1

    if scored == n:
        print(f"\n{PASS} stage_eval 완료 → 05_evaluate.py 실행 가능")
        return True
    return False


# ── Stage 5: Batch Eval (05_evaluate.py --result_files) ──────────────────
def stage_batch_eval(n_files: int = 2, n_samples: int = 2):
    """05_evaluate.py가 judge 1회 로드로 N개 파일을 batch 처리하는지 검증."""
    header(f"Stage: batch_eval — {n_files}개 파일을 judge 1회 로드로 평가")
    import tempfile, subprocess, json, gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tmpdir = tempfile.mkdtemp(prefix="smoke_batch_eval_")
    files = []
    for i in range(n_files):
        d = Path(tmpdir) / f"fake_model_{i}"
        d.mkdir(parents=True)
        f = d / "gold_results.jsonl"
        with open(f, "w", encoding="utf-8") as fp:
            for j in range(n_samples):
                fp.write(json.dumps({
                    "idx": j, "sid": -1,
                    "generated": "SpO2 93% → O2 보충 유지.",
                }, ensure_ascii=False) + "\n")
        files.append(str(f))

    cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "pipeline" / "05_evaluate.py"),
           "--result_files"] + files
    result = ok("batch eval subprocess", lambda: subprocess.run(cmd, check=True, capture_output=True, text=True))
    if result is None:
        return False

    # judge 1회 로드 확인 — "Judge 모델 로드" 라인이 정확히 1회만 출력
    load_lines = [l for l in result.stdout.splitlines() if "Judge 모델 로드" in l]
    print(f"  Judge 로드 횟수: {len(load_lines)} (예상 1)")
    if len(load_lines) != 1:
        print(f"  {FAIL} judge가 {len(load_lines)}회 로드됨 — batch 아님")
        return False

    print(f"\n{PASS} stage_batch_eval 완료 → 05_evaluate.py --result_files 정상")
    return True


# ── Stage 6: Orchestrator (run_all.py 2단계 흐름) ────────────────────────
def stage_orchestrator():
    """run_all.py의 Phase1(병렬 학습+추론) + Phase2(batch eval) 흐름을 mock subprocess로 검증."""
    header("Stage: orchestrator — run_all.py 2단계 흐름 (GPU 불필요)")
    import importlib.util, threading, time, tempfile, shutil
    from unittest.mock import patch

    spec = importlib.util.spec_from_file_location(
        "run_all", Path(__file__).resolve().parent.parent / "run_all.py"
    )
    run_all = importlib.util.module_from_spec(spec)
    ok("run_all 로드", lambda: spec.loader.exec_module(run_all))

    # 임시 출력 디렉토리로 INFER_OUT/EVAL_OUT 우회
    tmpbase = Path(tempfile.mkdtemp(prefix="smoke_orch_"))
    orig_infer = run_all.INFER_OUT
    orig_eval = run_all.EVAL_OUT
    run_all.INFER_OUT = tmpbase / "infer"
    run_all.EVAL_OUT = tmpbase / "eval"

    calls = []
    call_lock = threading.Lock()

    def fake_run_cmd(cmd, desc, gpus, tag=""):
        with call_lock:
            calls.append((time.time(), desc, gpus, tag, list(cmd)))
        time.sleep(0.1)
        # infer 호출 후 결과 파일 생성 (Phase 2가 픽업하려면 필요)
        cmd_str = " ".join(str(c) for c in cmd)
        if "04_inference.py" in cmd_str:
            ix = cmd.index("--out_tag")
            out_tag = cmd[ix + 1]
            f = run_all.INFER_OUT / out_tag / "gold_results.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('{"idx":0,"sid":-1,"generated":"x"}\n')
        elif "05_evaluate.py" in cmd_str:
            # batch eval 출력 파일도 생성 (make_sample_md용)
            idx_rf = cmd.index("--result_files")
            for rf in cmd[idx_rf + 1:]:
                tag_name = Path(rf).parent.name
                out_f = run_all.EVAL_OUT / tag_name / "gold_results_scores.jsonl"
                out_f.parent.mkdir(parents=True, exist_ok=True)
                out_f.write_text('{"idx":0,"brevity_score":3,"critical_score":3,"sum_score":6,"sid":-1,"generated":"x"}\n')
        return True

    try:
        with patch.object(run_all, "run_cmd", fake_run_cmd), \
             patch.object(run_all, "make_sample_md", lambda *a, **k: None):
            run_all.run_parallel(
                models=["llama", "qwen"],
                gpus_str="4,5,6,7",
                gpus_per_job=2,
                skip_done=False,
                only_eval=False,
                exps=[("raw", None, None, None)],
                eval_gpus="4,5,6,7",
            )
    finally:
        run_all.INFER_OUT = orig_infer
        run_all.EVAL_OUT = orig_eval
        shutil.rmtree(tmpbase, ignore_errors=True)

    infer_calls = [c for c in calls if "Inference" in c[1]]
    eval_calls = [c for c in calls if "Evaluate" in c[1]]

    print(f"  Phase 1 — 추론 호출: {len(infer_calls)}회 (예상 2: llama+qwen)")
    print(f"  Phase 2 — eval 호출:  {len(eval_calls)}회 (예상 1: batch)")

    if len(eval_calls) != 1:
        print(f"  {FAIL} eval이 batch로 묶이지 않음 ({len(eval_calls)}회)")
        return False
    if len(infer_calls) != 2:
        print(f"  {FAIL} 추론 호출 횟수 불일치")
        return False

    last_infer_t = max(c[0] for c in infer_calls) if infer_calls else 0
    if eval_calls[0][0] < last_infer_t:
        print(f"  {FAIL} Phase 분리 깨짐 (eval이 infer 완료 전 시작)")
        return False

    if "--result_files" not in eval_calls[0][4]:
        print(f"  {FAIL} eval이 --result_files 안 씀")
        return False

    # GPU 충돌 검사 — eval 동안 다른 잡이 같은 GPU 점유했는지
    eval_start, eval_end = eval_calls[0][0], eval_calls[0][0] + 0.1
    eval_gpus_set = set(eval_calls[0][2].split(","))
    for t, desc, gpus, tag, _ in calls:
        if desc == eval_calls[0][1]:
            continue
        if eval_start <= t <= eval_end:
            overlap = eval_gpus_set & set(gpus.split(","))
            if overlap:
                print(f"  {FAIL} eval 중 {tag}/{desc}가 GPU {overlap} 사용")
                return False

    print(f"  {PASS} Phase 1 (병렬) → Phase 2 (batch eval) 순서/GPU 모두 정상")
    return True


# ── main ──────────────────────────────────────────────────────────────────
ALL_MODELS = ["llama", "qwen", "gemma4", "gemma4_31b", "qwen35", "hari"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="파이프라인 단계별 smoke test")
    parser.add_argument("--stage", choices=["data", "sft", "rlaif", "infer", "eval", "batch_eval", "orchestrator"], default=None)
    parser.add_argument("--all", action="store_true", help="모든 단계 순서대로 실행")
    parser.add_argument(
        "--models", nargs="+",
        choices=ALL_MODELS + ["all"], default=["llama"],
        help="테스트할 모델 (여러 개 가능, 'all'로 전체 선택)",
    )
    parser.add_argument("--loss", choices=["dpo", "simpo", "both"], default="both",
                        help="rlaif loss (기본 both — dpo와 simpo 둘 다)")
    parser.add_argument("--steps", type=int, default=3, help="학습 steps (sft/rlaif용)")
    parser.add_argument("--n", type=int, default=2, help="샘플 수 (infer/eval용)")
    parser.add_argument("--samples", type=int, default=8, help="학습용 데이터 샘플 수")
    parser.add_argument("--gpus", type=str, default=None)
    args = parser.parse_args()

    models = ALL_MODELS if "all" in args.models else args.models

    if not (args.stage or args.all):
        parser.print_help()
        sys.exit(0)

    # {model: {stage: bool}}
    results: dict = {}

    # data 단계는 모델 무관 — 한 번만
    if args.all or args.stage == "data":
        stage_data()
        results.setdefault("_common", {})["data"] = True

    for model in models:
        results.setdefault(model, {})

        if args.all or args.stage == "sft":
            results[model]["sft"] = stage_sft(model, args.steps, args.samples)

        if args.all or args.stage == "rlaif":
            losses = ["dpo", "simpo"] if args.loss == "both" else [args.loss]
            for loss in losses:
                results[model][f"rlaif_{loss}"] = stage_rlaif(model, loss, args.steps, args.samples)

        if args.all or args.stage == "infer":
            results[model]["infer"] = stage_infer(model, args.n)

    # eval / batch_eval / orchestrator 단계는 모델 무관 — 한 번만
    if args.all or args.stage == "eval":
        results.setdefault("_common", {})["eval"] = stage_eval_inline(args.n)

    if args.all or args.stage == "batch_eval":
        results.setdefault("_common", {})["batch_eval"] = stage_batch_eval(n_files=2, n_samples=args.n)

    if args.all or args.stage == "orchestrator":
        results.setdefault("_common", {})["orchestrator"] = stage_orchestrator()

    # ── 결과 요약 테이블 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" Smoke Test 결과 요약")
    print(f"{'='*60}")

    common = results.pop("_common", {})
    for stage, flag in common.items():
        print(f"  {'✓' if flag else '✗'} [공통] {stage}")

    if results:
        model_stages = sorted({s for v in results.values() for s in v})
        header_row = f"  {'모델':<12}" + "".join(f" {s:<8}" for s in model_stages)
        print(header_row)
        print(f"  {'-'*12}" + "".join(f" {'-'*8}" for _ in model_stages))
        for model, stages in results.items():
            row = f"  {model:<12}" + "".join(
                f" {'✓' if stages.get(s) else '✗':<8}" for s in model_stages
            )
            print(row)
    print(f"{'='*60}")
