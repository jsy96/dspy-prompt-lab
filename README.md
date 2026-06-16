# dspy-prompt-lab

一个部署到 **Vercel** 的网页，用真正的 Python **DSPy** 做"提示词优化 + 效果对比"。

- **功能①**：给若干「输入 → 期望输出」样例，后端用 `dspy.BootstrapFewShot` 自动生成一个**优化后的提示词**（指令 + 少量带推理的 few-shot 演示）。
- **功能②**：把"优化提示词"和"普通提示词"分别喂给**同一个 DeepSeek**，在同一个测试输入上对比输出差异（含逐词 diff）。

> 为什么是 Vercel 不是 EdgeOne：EdgeOne 边缘函数只能跑 JavaScript，无法 `import dspy`。Vercel 支持 Python Serverless Functions，能直接跑真正的 dspy-glm。

## 架构

```
浏览器 index.html (纯前端，无构建)
   │  fetch /api/*
   ▼
Vercel Python Function  api/app.py (Flask 单 app)
   ├── POST /api/optimize   dspy BootstrapFewShot 生成优化提示词
   ├── POST /api/compare    普通 vs 优化 提示词 喂 DeepSeek 对比
   └── GET  /api/health     存活探测（不调 API）
        │
        └──→ DeepSeek API  https://api.deepseek.com   [key 从环境变量读]
```

- **唯一环境变量**：`DEEPSEEK_API_KEY`。teacher（bootstrap）、裁判（judge）、对比模型都用 DeepSeek，复用同一个 key。
- **模型**：`deepseek-v4-flash`（DeepSeek-V4-Flash 非思考模式，2026-04-24 起可用）。旧别名 `deepseek-chat` 当前仍路由到同一模型，但官方将于 **2026/07/24 15:59 UTC** 彻底停用（届时直接报错），因此本项目按官方"新接入直接用新名"的建议直接采用 `deepseek-v4-flash`。

## 文件结构

```
dspy-prompt-lab/
├── api/
│   ├── app.py          # Flask app：/api/optimize、/api/compare、/api/health
│   └── dspy_lab.py     # 共享：DeepSeek LM 工厂、关磁盘缓存、Signature 构造
├── index.html          # 单文件前端（内联 CSS+JS）
├── requirements.txt    # dspy==3.2.1, flask>=3.0
├── vercel.json         # maxDuration=300 + /api/* rewrite
├── .env.example        # 本地调试用（拷成 .env 填 key）
└── README.md
```

## 部署到 Vercel

### 方式 A：Git 集成（推荐）
1. 推到 GitHub 仓库（`.env` 已被 `.gitignore` 排除，不会泄露 key）。
2. Vercel 控制台 → **New Project** → Import 该仓库 → Framework Preset 选 **Other**（Build / Output 都留空）。**切勿选 Flask preset**：它会把 app 打包成 single function，导致 `vercel.json` 里 `functions["api/app.py"]` 匹配不上，构建报 `unmatched-function-pattern`；只有 Other（file-based function）下 `api/app.py` 才被识别为函数，`maxDuration` 等配置才生效。
3. **Settings → Environment Variables** 添加 `DEEPSEEK_API_KEY` = 你的 key，三个环境（Production / Preview / Development）都勾上。
4. Deploy。失败先看 **Build Logs**（pip 体积）与 **Function Logs**（dspy import / 超时）。

### 方式 B：Vercel CLI
```bash
npm i -g vercel
cd dspy-prompt-lab
vercel link
vercel env add DEEPSEEK_API_KEY      # 粘贴 key，选所有环境
vercel --prod
```

## 本地调试

```bash
# 1) 把 key 拉到本地 .env（或手写 .env：DEEPSEEK_API_KEY=sk-...）
vercel env pull .env

# 2) 启动本地（自带 Python runtime + 静态托管）
vercel dev
# 打开提示的地址（通常 http://localhost:3000）
```

不想装 Vercel CLI 时，可直接用 Flask 跑后端（仅 API，无静态托管）：
```bash
# 本机用 uv（无系统 python）
uv venv .venv --python 3.12
uv pip install -r requirements.txt
# Windows (git bash)
DEEPSEEK_API_KEY=sk-xxx PYTHONPATH=api uv run --no-project python -c "from app import app; app.run(port=5000)"
```
然后浏览器开 `http://localhost:5000`（前端用 fetch 同源 `/api/*`；如需页面，单独用任意静态服务器托管 `index.html` 并把 fetch 改成绝对地址 `http://localhost:5000/api/...`）。

## 端到端验证

**一键自动化**（需 key）：`test_e2e.py` 用 Flask test client 跑通 optimize → compare 全链路。
```bash
export DEEPSEEK_API_KEY=sk-xxx   # Windows git bash
PYTHONPATH=api uv run --no-project --with 'dspy==3.2.1' --with 'flask>=3.0,<4.0' python test_e2e.py
```

**手动 curl**：
```bash
# 存活探测（不花钱）
curl http://localhost:3000/api/health

# 功能①：生成优化提示词
curl -X POST http://localhost:3000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"examples":[
        {"input":"Translate: good morning","output":"早上好"},
        {"input":"Translate: thank you","output":"谢谢"},
        {"input":"Translate: sorry","output":"对不起"}],
       "instruction":"Translate English phrases to Simplified Chinese."}'
# 期望：demos 非空（含 reasoning），template_text 含 system + 多轮演示对

# 功能②：对比（用上一步返回的 instruction + demos）
curl -X POST http://localhost:3000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"plain_prompt":"把下面的英文翻译成中文。",
       "instruction":"<上一步 instruction>",
       "demos":[ <上一步 demos> ],
       "test_input":"Translate: good night"}'
# 期望：plain.output 与 optimized.output 各一段 + latency
```

成功标志：`stats.bootstrapped >= 1` 说明 BootstrapFewShot 真的产出了带推理的演示。若 `demos=[]`，多半是裁判模型被 DeepSeek 限流（429），过一会重试或减少样例数。

## 关键实现点

- **关磁盘缓存**：`dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)`（[api/dspy_lab.py](api/dspy_lab.py)）。Vercel 文件系统只读，DSPy 默认写 `~/.dspy_cache` 会失败；用内存缓存兜底。
- **导出优化提示词**：编译后取 `compiled.named_predictors()[0][1]`，用 `ChatAdapter().format_system_message(sig)` + `format_demos(sig, demos)` 拼出完整提示词（system 指令 + few-shot 演示对）。
- **对比侧用真 dspy**：`/api/compare` 的"优化侧"重建 `dspy.Predict(input -> reasoning, output)` 并挂上演示，让 dspy 自动渲染完整优化提示词 + 解析输出，而非手工拼字符串——这样对比才真实反映优化效果。
- **超时控制**：样例 ≤8、few-shot ≤3、`max_rounds=1`、`max_errors=3`。最坏 ~16 次 DeepSeek 调用（teacher + judge），P95 ~128s，远低于 Vercel Hobby 的 300s 上限。

## 包大小

Vercel Python 函数上限 **500MB**（unzipped）。实测 `dspy 3.2.1 + flask + litellm` 全量依赖 **160MB**（litellm 55M / numpy 42M / openai 8M / tokenizers 7.5M / dspy 本体仅 1.4M），远低于上限，无需瘦身。若未来依赖膨胀超限，用精装：
```bash
pip install dspy==3.2.1 --no-deps
pip install litellm pydantic openai tenacity diskcache json-repair regex orjson cachetools cloudpickle flask
```

## 备注

- teacher 与裁判都走 DeepSeek，故全程只需一个 key。若想换回"dspy-glm"原意（用智谱 GLM 当 teacher），把 [api/dspy_lab.py](api/dspy_lab.py) 的 `make_lm` 改成智谱 anthropic 端点（需额外配 GLM key）。
- 模型用 `deepseek-v4-flash`（官方对"新接入"的推荐）。旧别名 `deepseek-chat` 将于 2026/07/24 15:59 UTC 停用，切换只需改 `DEEPSEEK_MODEL` 常量一处，base_url 与 key 均不变。
