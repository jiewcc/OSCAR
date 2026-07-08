#!/usr/bin/env python3
"""Run OSCAR paper benchmarks against an OpenAI-compatible server.

Supported tasks:
  gpqa      GPQA Diamond multiple-choice accuracy
  aime25    AIME 2025 exact integer accuracy
  math500   MATH-500 boxed-answer accuracy
  humaneval HumanEval pass@1 with local unit tests
  lcbv6     LiveCodeBench code_generation_lite test6 pass@1
"""

import argparse
import base64
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
import math
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
LCB_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"

GPQA_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}"""

MATH_TEMPLATE = """Solve the problem. Think step by step, and put only the final answer in \\boxed{{}}.

{problem}"""

AIME_TEMPLATE = """Solve the following AIME 2025 problem. The answer is an integer from 0 to 999. Think step by step, and put only the final integer answer in \\boxed{{}}.

{problem}"""

HUMANEVAL_TEMPLATE = """Complete the following Python function. Return only valid Python code, with no Markdown fences.

{prompt}"""

LCB_TEMPLATE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question_content}

{starter_block}
"""


def progress(items, *, total: int, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_file(root: Path | None, filename: str) -> Path | None:
    if root is None or not root.exists():
        return None
    direct = first_existing([root / filename, root / "datasets" / filename])
    if direct is not None:
        return direct
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def dataset_dir(data_root: Path | None, env_name: str, candidates: list[str]) -> Path | None:
    env_value = os.environ.get(env_name)
    if env_value:
        path = Path(env_value)
        if path.exists():
            return path
    if data_root is None:
        return None
    paths: list[Path] = []
    for candidate in candidates:
        paths.extend([data_root / candidate, data_root / "datasets" / candidate])
    return first_existing(paths)


def load_dataset_split(name_or_path: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The requested task needs the `datasets` package. Install it in the "
            "same Python environment used to run this script."
        ) from exc
    return load_dataset(name_or_path)[split]


def cached_download(url: str, cache_dir: Path, filename: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(url, path)
    return path


def strip_thinking(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def extract_boxed(text: str) -> str:
    text = strip_thinking(text)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text, flags=re.S)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"(?i)(?:final answer|answer)\s*[:：]\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def extract_int(text: str) -> str:
    boxed = extract_boxed(text)
    nums = re.findall(r"-?\d+", boxed)
    if nums:
        return str(int(nums[-1]))
    nums = re.findall(r"-?\d+", strip_thinking(text))
    return str(int(nums[-1])) if nums else ""


def extract_mc(text: str) -> str:
    text = strip_thinking(text)
    for pat in (
        r"(?i)answer\s*[:：]\s*([A-D])",
        r"(?i)final answer\s*[:：]?\s*([A-D])",
        r"\b([A-D])\b",
    ):
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].upper()
    return ""


def norm_math(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", s)
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\,", ""),
        ("\\!", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
    ):
        s = s.replace(old, new)
    return s.replace(" ", "").replace("\n", "").replace("$", "").lower().strip(".")


def math_equal(prediction: str, reference: str) -> bool:
    if norm_math(prediction) == norm_math(reference):
        return True
    repo = Path(__file__).resolve().parents[2]
    reasoning_dir = repo / "sglang-research" / "benchmark" / "reasoning_benchmark"
    if reasoning_dir.is_dir():
        sys.path.insert(0, str(reasoning_dir))
        try:
            from eval_utils import math_equal as symbolic_math_equal

            return bool(symbolic_math_equal(prediction, reference))
        except Exception:
            pass
    return False


def strip_code(text: str) -> str:
    text = strip_thinking(text)
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced[-1].strip()
    match = re.search(
        r"(?m)^(?:from\s+\S+\s+import\s+.*|import\s+.*|def\s+\w+\s*\(|class\s+Solution\b)",
        text,
    )
    return text[match.start() :].strip() if match else text.strip()


def stripped_lines(val: str) -> list[str]:
    return [line.strip() for line in val.strip().splitlines()]


def same_output(prediction: str, expected: str) -> bool:
    pred_lines = stripped_lines(prediction)
    exp_lines = stripped_lines(expected)
    if len(pred_lines) != len(exp_lines):
        return False
    for pred, exp in zip(pred_lines, exp_lines):
        if pred == exp:
            continue
        try:
            if [Decimal(x) for x in pred.split()] == [Decimal(x) for x in exp.split()]:
                continue
        except Exception:
            pass
        return False
    return True


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        system_message: str,
        api_key: str,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "The `openai` package is required to call the OpenAI-compatible "
                "SGLang endpoint. Install it in the Python environment used for eval."
            ) from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_message = system_message

    def ask(self, prompt: str) -> str:
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})
        for trial in range(9):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body={"top_k": self.top_k},
                )
                return resp.choices[0].message.content or ""
            except Exception:
                if trial == 8:
                    raise
                time.sleep(min(30, 2**trial))
        return ""


def load_gpqa(cache_dir: Path, data_root: Path | None) -> list[dict[str, Any]]:
    path = find_file(data_root, "gpqa_diamond.csv")
    if path is None:
        path = cached_download(GPQA_URL, cache_dir, "gpqa_diamond.csv")
    rows = list(csv.DictReader(path.read_text().splitlines()))
    rng = random.Random(0)
    samples = []
    for row in rows:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        perm = rng.sample(range(4), 4)
        shuffled = [choices[i] for i in perm]
        samples.append(
            {
                "id": str(len(samples)),
                "prompt": GPQA_TEMPLATE.format(
                    Question=row["Question"],
                    A=shuffled[0],
                    B=shuffled[1],
                    C=shuffled[2],
                    D=shuffled[3],
                ),
                "answer": "ABCD"[perm.index(0)],
            }
        )
    return samples


def load_aime25(data_root: Path | None) -> list[dict[str, Any]]:
    source = dataset_dir(data_root, "AIME25_DIR", ["aime25", "math-ai/aime25", "math-ai--aime25"])
    ds = load_dataset_split(str(source) if source else "math-ai/aime25", "test")
    return [
        {
            "id": str(row.get("id", i)),
            "prompt": AIME_TEMPLATE.format(problem=row["problem"]),
            "answer": str(row["answer"]),
        }
        for i, row in enumerate(ds)
    ]


def load_math500(data_root: Path | None) -> list[dict[str, Any]]:
    source = dataset_dir(
        data_root,
        "MATH500_DIR",
        ["MATH-500", "math500", "HuggingFaceH4/MATH-500", "HuggingFaceH4--MATH-500"],
    )
    ds = load_dataset_split(str(source) if source else "HuggingFaceH4/MATH-500", "test")
    return [
        {
            "id": row.get("unique_id", str(i)),
            "prompt": MATH_TEMPLATE.format(problem=row["problem"]),
            "answer": row["answer"],
        }
        for i, row in enumerate(ds)
    ]


def load_humaneval(data_root: Path | None) -> list[dict[str, Any]]:
    source = dataset_dir(
        data_root,
        "HUMANEVAL_DIR",
        ["openai_humaneval", "openai/openai_humaneval", "openai--openai_humaneval"],
    )
    return [dict(row) for row in load_dataset_split(str(source) if source else "openai/openai_humaneval", "test")]


def translate_private_test_cases(encoded_data: str) -> list[dict[str, str]]:
    decoded = base64.b64decode(encoded_data)
    decompressed = zlib.decompress(decoded)
    return json.loads(pickle.loads(decompressed))


def load_lcbv6(cache_dir: Path, data_root: Path | None) -> list[dict[str, Any]]:
    path = find_file(data_root, "test6.jsonl")
    if path is None:
        path = cached_download(LCB_URL, cache_dir, "lcb_test6.jsonl")
    samples = []
    for i, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        public_tests = json.loads(row["public_test_cases"])
        private_tests = translate_private_test_cases(row["private_test_cases"])
        starter_code = row.get("starter_code") or ""
        if starter_code:
            starter_block = (
                "Use this starter code and return the complete Python solution.\n"
                f"```python\n{starter_code}\n```"
            )
        else:
            starter_block = (
                "Read from stdin and write to stdout. Return the complete Python program.\n"
                "```python\n# YOUR CODE HERE\n```"
            )
        metadata = json.loads(row.get("metadata") or "{}")
        samples.append(
            {
                "id": row.get("question_id", str(i)),
                "prompt": LCB_TEMPLATE.format(
                    question_content=row["question_content"],
                    starter_block=starter_block,
                ),
                "tests": public_tests + private_tests,
                "fn_name": metadata.get("func_name"),
            }
        )
    return samples


def check_humaneval(sample: dict[str, Any], completion: str, timeout: int) -> tuple[bool, str]:
    code = strip_code(completion)
    entry = sample["entry_point"]
    full = code if f"def {entry}" in code else sample["prompt"] + "\n" + code
    program = full + "\n" + sample["test"] + f"\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "check.py"
        path.write_text(program)
        try:
            res = subprocess.run(
                [sys.executable, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return res.returncode == 0, (res.stderr or res.stdout)[-1000:]


def check_lcb(sample: dict[str, Any], completion: str, timeout: int) -> tuple[bool, str]:
    code = strip_code(completion)
    if not code:
        return False, "empty_code"
    fn_name = sample.get("fn_name")
    with tempfile.TemporaryDirectory() as tmpdir:
        if fn_name:
            wrapper = Path(tmpdir) / "check.py"
            wrapper.write_text(
                "import json, sys\n"
                + code
                + "\n"
                + "fn_name = sys.argv[1]\n"
                + "raw_inputs = sys.stdin.read().splitlines()\n"
                + "args = [json.loads(x) for x in raw_inputs if x.strip()]\n"
                + "target = Solution() if 'Solution' in globals() else globals()\n"
                + "fn = getattr(target, fn_name) if not isinstance(target, dict) else target[fn_name]\n"
                + "print(json.dumps(fn(*args), separators=(',', ':')))\n"
            )
            for test in sample["tests"]:
                try:
                    res = subprocess.run(
                        [sys.executable, str(wrapper), fn_name],
                        input=test["input"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout,
                        cwd=tmpdir,
                    )
                except subprocess.TimeoutExpired:
                    return False, "timeout"
                if res.returncode != 0:
                    return False, (res.stderr or res.stdout)[-1000:]
                if not same_output(res.stdout, test["output"]):
                    return False, f"wrong_answer stdout={res.stdout[-200:]!r} expected={test['output'][-200:]!r}"
            return True, ""

        program = Path(tmpdir) / "solution.py"
        program.write_text(code)
        for test in sample["tests"]:
            try:
                res = subprocess.run(
                    [sys.executable, str(program)],
                    input=test["input"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return False, "timeout"
            if res.returncode != 0:
                return False, (res.stderr or res.stdout)[-1000:]
            if not same_output(res.stdout, test["output"]):
                return False, f"wrong_answer stdout={res.stdout[-200:]!r} expected={test['output'][-200:]!r}"
    return True, ""


def load_samples(task: str, cache_dir: Path, data_root: Path | None) -> list[dict[str, Any]]:
    if task == "gpqa":
        return load_gpqa(cache_dir, data_root)
    if task == "aime25":
        return load_aime25(data_root)
    if task == "math500":
        return load_math500(data_root)
    if task == "humaneval":
        return load_humaneval(data_root)
    if task == "lcbv6":
        return load_lcbv6(cache_dir, data_root)
    raise ValueError(f"unsupported task {task}")


def score_one(args: argparse.Namespace, sample: dict[str, Any], client: ChatClient) -> dict[str, Any]:
    prompt = HUMANEVAL_TEMPLATE.format(prompt=sample["prompt"]) if args.task == "humaneval" else sample["prompt"]
    response = client.ask(prompt)
    detail = ""
    if args.task == "gpqa":
        prediction = extract_mc(response)
        correct = prediction == sample["answer"]
    elif args.task == "aime25":
        prediction = extract_int(response)
        correct = prediction == str(int(sample["answer"]))
    elif args.task == "math500":
        prediction = extract_boxed(response)
        correct = math_equal(prediction, sample["answer"])
    elif args.task == "humaneval":
        correct, detail = check_humaneval(sample, response, args.code_timeout)
        prediction = "pass" if correct else "fail"
    elif args.task == "lcbv6":
        correct, detail = check_lcb(sample, response, args.code_timeout)
        prediction = "pass" if correct else "fail"
    else:
        raise ValueError(args.task)
    return {
        "id": sample.get("id") or sample.get("task_id"),
        "answer": sample.get("answer"),
        "prediction": prediction,
        "correct": bool(correct),
        "detail": detail,
        "response_chars": len(response),
        "response": response,
    }


def summarize(task: str, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    scores = [1.0 if row["correct"] else 0.0 for row in rows]
    chars = [float(row["response_chars"]) for row in rows]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_chars = sum(chars) / len(chars) if chars else 0.0
    score_std = math.sqrt(sum((x - mean_score) ** 2 for x in scores) / len(scores)) if scores else 0.0
    chars_std = math.sqrt(sum((x - mean_chars) ** 2 for x in chars) / len(chars)) if chars else 0.0
    return {
        "task": task,
        "num_examples": len(rows),
        "correct": int(sum(scores)),
        "score": mean_score,
        "score:std": score_std,
        "score:stderr": score_std / math.sqrt(len(scores)) if scores else 0.0,
        "chars": mean_chars,
        "chars:std": chars_std,
        "elapsed": elapsed,
    }


def write_eval_log(task: str, model: str, metrics: dict[str, Any], outdir: Path) -> None:
    lines = [
        f"Evaluation results for {task} on {model}",
        "=" * 100,
        "+" + "-" * 20 + "+" + "-" * 24 + "+",
        "|       Metric         |         Value          |",
        "+" + "-" * 20 + "+" + "-" * 24 + "+",
    ]
    for key in ("chars", "chars:std", "score", "score:std", "score:stderr"):
        value = metrics[key]
        lines.append(f"|   {task}/{key:<14s} | {float(value):>22.6f} |")
    lines.append("+" + "-" * 20 + "+" + "-" * 24 + "+")
    lines.append(f"(elapsed: {metrics['elapsed']:.1f}s)")
    text = "\n".join(lines) + "\n"
    (outdir / "eval.log").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["gpqa", "aime25", "math500", "humaneval", "lcbv6"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--code-timeout", type=int, default=8)
    parser.add_argument("--system-message", default="")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root) if args.data_root else None
    samples = load_samples(args.task, outdir / "_cache", data_root)
    if args.num_examples is not None:
        samples = samples[: args.num_examples]

    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        system_message=args.system_message,
        api_key=args.api_key,
    )
    print(f"=== running {args.task} eval ===", flush=True)
    print(f"  model={args.model} base_url={args.base_url}")
    print(f"  examples={len(samples)} threads={args.num_threads} max_tokens={args.max_tokens}")

    start = time.time()
    rows = []
    output_path = outdir / f"{args.task}.jsonl"
    with output_path.open("w") as out:
        if args.num_threads <= 1:
            iterator = (score_one(args, sample, client) for sample in samples)
            for row in progress(iterator, total=len(samples), desc=args.task):
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                futures = [executor.submit(score_one, args, sample, client) for sample in samples]
                for fut in progress(as_completed(futures), total=len(futures), desc=args.task):
                    row = fut.result()
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()

    metrics = summarize(args.task, rows, time.time() - start)
    metrics["output"] = str(output_path)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_eval_log(args.task, args.model, metrics, outdir)


if __name__ == "__main__":
    main()
