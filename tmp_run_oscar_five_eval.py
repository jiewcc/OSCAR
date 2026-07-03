#!/usr/bin/env python3
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import json
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

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

GPQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
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
LCB_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl"
LCB_TEMPLATE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {question_content}

{starter_block}
"""


def norm_math(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\,", ""), ("\\!", "")]:
        s = s.replace(a, b)
    s = s.replace(" ", "").replace("\n", "").replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.lower().strip(".")


def extract_boxed(text: str) -> str:
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
    nums = re.findall(r"-?\d+", text)
    return str(int(nums[-1])) if nums else ""


def extract_mc(text: str) -> str:
    for pat in [
        r"(?i)answer\s*[:：]\s*([A-D])",
        r"(?i)final answer\s*[:：]?\s*([A-D])",
        r"\b([A-D])\b",
    ]:
        m = re.findall(pat, text)
        if m:
            return m[-1].upper()
    return ""


def strip_code(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m[-1].strip()
    m = re.search(r"(?m)^(?:from\s+\S+\s+import\s+.*|import\s+.*|def\s+\w+\s*\(|class\s+Solution\b)", text)
    if m:
        return text[m.start() :].strip()
    return text.strip()


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


class Client:
    def __init__(self, base_url: str, model: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def ask(self, prompt: str) -> str:
        for i in range(8):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body={"top_k": self.top_k},
                )
                return resp.choices[0].message.content or ""
            except Exception:
                if i == 7:
                    raise
                time.sleep(min(30, 2**i))
        return ""


def load_gpqa() -> list[dict[str, Any]]:
    df = pd.read_csv(GPQA_URL)
    rng = random.Random(0)
    out = []
    for _, row in df.iterrows():
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        perm = rng.sample(range(4), 4)
        shuffled = [choices[i] for i in perm]
        answer = "ABCD"[perm.index(0)]
        out.append(
            {
                "id": str(len(out)),
                "prompt": GPQA_TEMPLATE.format(
                    Question=row["Question"], A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3]
                ),
                "answer": answer,
            }
        )
    return out


def load_aime() -> list[dict[str, Any]]:
    ds = load_dataset("math-ai/aime25")["test"]
    return [
        {"id": str(x.get("id", i)), "prompt": AIME_TEMPLATE.format(problem=x["problem"]), "answer": str(x["answer"])}
        for i, x in enumerate(ds)
    ]


def load_math500() -> list[dict[str, Any]]:
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    return [
        {"id": x.get("unique_id", str(i)), "prompt": MATH_TEMPLATE.format(problem=x["problem"]), "answer": x["answer"]}
        for i, x in enumerate(ds)
    ]


def load_humaneval() -> list[dict[str, Any]]:
    return [dict(x) for x in load_dataset("openai/openai_humaneval")["test"]]


def translate_private_test_cases(encoded_data: str) -> list[dict[str, str]]:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def load_lcbv6(cache_dir: Path) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "test6.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(LCB_URL, path)
    samples = []
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            public_tests = json.loads(row["public_test_cases"])
            private_tests = translate_private_test_cases(row["private_test_cases"])
            starter_code = row.get("starter_code") or ""
            if starter_code:
                starter_block = (
                    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
                    f"```python\n{starter_code}\n```\n"
                )
            else:
                starter_block = (
                    "Read the inputs from stdin, solve the problem, and write the answer to stdout. "
                    "Enclose your code within delimiters as follows.\n"
                    "```python\n# YOUR CODE HERE\n```"
                )
            metadata = json.loads(row.get("metadata") or "{}")
            samples.append(
                {
                    "id": row.get("question_id", str(i)),
                    "prompt": LCB_TEMPLATE.format(question_content=row["question_content"], starter_block=starter_block),
                    "tests": public_tests + private_tests,
                    "fn_name": metadata.get("func_name"),
                }
            )
    return samples


def check_humaneval(sample: dict[str, Any], completion: str, timeout: int = 8) -> tuple[bool, str]:
    code = strip_code(completion)
    full = code if ("def " + sample["entry_point"] in code) else sample["prompt"] + "\n" + code
    program = full + "\n" + sample["test"] + f"\ncheck({sample['entry_point']})\n"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "check.py"
        path.write_text(program)
        try:
            res = subprocess.run(
                [sys.executable, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return res.returncode == 0, (res.stderr or res.stdout)[-1000:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def check_lcb(sample: dict[str, Any], completion: str, timeout: int = 6) -> tuple[bool, str]:
    code = strip_code(completion)
    if not code:
        return False, "empty_code"
    fn_name = sample.get("fn_name")
    with tempfile.TemporaryDirectory() as td:
        if fn_name:
            wrapper = Path(td) / "check.py"
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
                        cwd=td,
                    )
                except subprocess.TimeoutExpired:
                    return False, "timeout"
                if res.returncode != 0:
                    return False, (res.stderr or res.stdout)[-1000:]
                if not same_output(res.stdout, test["output"]):
                    return False, f"wrong_answer stdout={res.stdout[-200:]!r} expected={test['output'][-200:]!r}"
            return True, ""

        program = Path(td) / "solution.py"
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
                    cwd=td,
                )
            except subprocess.TimeoutExpired:
                return False, "timeout"
            if res.returncode != 0:
                return False, (res.stderr or res.stdout)[-1000:]
            if not same_output(res.stdout, test["output"]):
                return False, f"wrong_answer stdout={res.stdout[-200:]!r} expected={test['output'][-200:]!r}"
    return True, ""


def eval_task(args: argparse.Namespace) -> None:
    client = Client(args.base_url, args.model, args.max_tokens, args.temperature, args.top_p, args.top_k)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.task == "gpqa":
        samples = load_gpqa()
    elif args.task == "aime25":
        samples = load_aime()
    elif args.task == "math500":
        samples = load_math500()
    elif args.task == "humaneval":
        samples = load_humaneval()
    elif args.task == "lcbv6":
        samples = load_lcbv6(Path(args.output_dir) / "_lcb_cache")
    else:
        raise SystemExit(f"unsupported task {args.task}")
    if args.num_examples:
        samples = samples[: args.num_examples]

    def run_one(sample: dict[str, Any]) -> dict[str, Any]:
        prompt = HUMANEVAL_TEMPLATE.format(prompt=sample["prompt"]) if args.task == "humaneval" else sample["prompt"]
        response = client.ask(prompt)
        if args.task == "gpqa":
            prediction = extract_mc(response)
            ok = prediction == sample["answer"]
        elif args.task == "aime25":
            prediction = extract_int(response)
            ok = prediction == str(int(sample["answer"]))
        elif args.task == "math500":
            prediction = extract_boxed(response)
            ok = norm_math(prediction) == norm_math(sample["answer"])
        elif args.task == "humaneval":
            ok, _ = check_humaneval(sample, response)
            prediction = "pass" if ok else "fail"
        else:
            ok, detail = check_lcb(sample, response)
            prediction = "pass" if ok else detail
        return {
            "id": sample.get("id") or sample.get("task_id"),
            "answer": sample.get("answer"),
            "prediction": prediction,
            "correct": bool(ok),
            "response": response,
        }

    correct = 0
    path = outdir / f"{args.task}.jsonl"
    with path.open("w") as f:
        if args.num_threads <= 1:
            iterator = (run_one(sample) for sample in samples)
            for row in tqdm(iterator, total=len(samples), desc=args.task):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                correct += bool(row["correct"])
        else:
            with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
                futs = [ex.submit(run_one, sample) for sample in samples]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=args.task):
                    row = fut.result()
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    correct += bool(row["correct"])
    metrics = {
        "task": args.task,
        "num_examples": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples) if samples else 0.0,
        "output": str(path),
    }
    (outdir / f"{args.task}.metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["gpqa", "humaneval", "aime25", "math500", "lcbv6"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--num-threads", type=int, default=1)
    eval_task(parser.parse_args())


if __name__ == "__main__":
    main()
