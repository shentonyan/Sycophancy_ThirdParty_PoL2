"""评分者后端。只用标准库，不需要额外依赖。

  ollama   本地 Ollama（默认 http://localhost:11434）
  openai   任何 OpenAI 兼容接口（DeepSeek、通义千问、智谱、OpenAI……）
  mock     模拟评分者，只用来检查流程能否跑通，结果没有任何实证意义
"""
import json, os, random, re, time, urllib.request


def _post(url, payload, headers=None, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_choice(text):
    m = re.search(r"\b([AB])\b", (text or "").strip().upper())
    return m.group(1) if m else None


class OllamaJudge:
    def __init__(self, model, host="http://localhost:11434", seed=0):
        self.model, self.host, self.seed = model, host.rstrip("/"), seed
        self.name = f"ollama:{model}"

    def ask(self, prompt):
        out = _post(f"{self.host}/api/chat", {
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": self.seed, "num_predict": 8},
        })
        return out["message"]["content"]


class OpenAIJudge:
    def __init__(self, model, base_url, api_key_env="OPENAI_API_KEY"):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")
        if not self.key:
            raise SystemExit(f"没有找到环境变量 {api_key_env}，请先设置 API Key。")
        self.name = f"openai:{model}"

    def ask(self, prompt):
        out = _post(f"{self.base_url}/chat/completions", {
            "model": self.model, "temperature": 0, "max_tokens": 8,
            "messages": [{"role": "user", "content": prompt}],
        }, headers={"Authorization": f"Bearer {self.key}"})
        return out["choices"][0]["message"]["content"]


class MockJudge:
    """按预设概率偏向谄媚回答的假评分者。仅用于流程检查。"""
    RATES = {"quality": 0.35, "user_pref": 0.7, "third_party": 0.2, "placebo": 0.33, "accuracy": 0.12}

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.name = "mock"

    def ask(self, prompt, condition=None, context=None, sycophantic_first=None):
        p = self.RATES[condition] if context == "wrong_suggestion" else 0.08
        pick_s = self.rng.random() < p
        if self.rng.random() < 0.1:          # 加一点位置偏好，检验两种顺序抵消是否有效
            return "A"
        return ("A" if sycophantic_first else "B") if pick_s else ("B" if sycophantic_first else "A")


def make_judge(args):
    if args.backend == "ollama":
        return OllamaJudge(args.model, host=args.host)
    if args.backend == "openai":
        return OpenAIJudge(args.model, args.base_url, args.api_key_env)
    return MockJudge(seed=args.seed)


def ask_with_retry(judge, prompt, retries=3, **mock_kw):
    last = None
    for i in range(retries):
        try:
            text = judge.ask(prompt, **mock_kw) if isinstance(judge, MockJudge) else judge.ask(prompt)
            choice = parse_choice(text)
            if choice:
                return choice, text
            last = text
        except Exception as e:  # 网络或服务端错误：稍等后重试
            last = f"ERROR: {e}"
            time.sleep(2 * (i + 1))
    return None, last
