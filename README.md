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
   ├── POST /api/compare    普通 vs 优化 提示词 喂当前 LLM 对比
   └── GET  /api/health     存活探测（不调 API）
        │
        └──→ 当前 LLM  GLM_API_KEY 在场→GLM-5.1 (https://open.bigmodel.cn/api/anthropic)
                否则→DeepSeek (https://api.deepseek.com)   [key 从环境变量读，见 _select_provider()]
```

- **环境变量**：本地用 `GLM_API_KEY`（智谱 token，走 GLM-5.1 的 anthropic 端点）；若只设 `DEEPSEEK_API_KEY` 则自动回退到 DeepSeek。Vercel 部署只配 `DEEPSEEK_API_KEY`。teacher（bootstrap）、裁判（judge）、对比模型三者复用同一个被选中的 provider + key。
- **模型**：本地默认 `glm-5.1`（`GLM_API_KEY` 命中时，端点 https://open.bigmodel.cn/api/anthropic）；回退 / Vercel 线上为 `deepseek-v4-flash`（DeepSeek-V4-Flash 非思考模式，2026-04-24 起可用）。旧别名 `deepseek-chat` 将于 **2026/07/24 15:59 UTC** 彻底停用。

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

## 本地部署（GLM-5.1）

本仓库已经把后端改造成**按环境变量自动选择模型**的双 provider 结构：同一个 `api/app.py` 既能本地跑 GLM-5.1，也能在 Vercel 上跑 DeepSeek。本节只讲本地用 GLM-5.1 的流程。

### 前置条件

- **uv**：本机没有系统级 Python，所有 Python 一律走 uv（PATH 里的 `uv`，或 `C:\ProgramData\chocolatey\bin\uv.exe`）。`start.bat` 会自动定位 uv，找不到就报错退出。
- **GLM_API_KEY**：一个智谱 token（和 Claude Code 在本机调用的 `ANTHROPIC_AUTH_TOKEN` 是同一个值）。无需另装系统 Python、无需 venv、无需 `pip install`。

### 三种设置 GLM_API_KEY 的方式

1. **临时设置**（只在当前 cmd 窗口有效）：在启动前先 `set GLM_API_KEY=你的智谱token`，再运行 `start.bat`。
2. **`.env` 文件**（推荐，最省事）：在仓库根目录建一个 `.env`，写入一行：
   ```env
   GLM_API_KEY=你的智谱token
   ```
   `.env` 的格式是 `KEY=VALUE`，`#` 开头为注释。`.env` 已被 `.gitignore` 排除，不会泄露 key。
3. **永久系统环境变量**：在 Windows 系统环境变量里设置 `GLM_API_KEY`，新开的 cmd 窗口都会带上。

> 注意：`.env` 里的值**不会覆盖已经存在的环境变量**（沿用 dotenv 语义，`start.bat` 里的判断是 `if not defined %%a set "%%a=%%b"`）。所以临时 set 优先级最高。

### 启动

双击根目录的 `start.bat`，或在 cmd 里运行：

```bat
start.bat
```

`start.bat` 会依次：定位 uv → 加载 `.env`（不覆盖已有变量）→ 检查 `GLM_API_KEY` 或 `DEEPSEEK_API_KEY` 至少有一个 → 打印当前用的是哪个 provider → 以 `PYTHONPATH=api` 执行：

```bat
uv run --no-project --with "dspy==3.2.1" --with "flask>=3.0,<4.0" python run_local.py
```

### 打开的地址

```
http://127.0.0.1:5000
```

这一个 URL 同时托管前端页面和所有 `/api/*` 接口——`run_local.py` 启动的 Flask 服务新增了 `GET /`，会用 `send_from_directory` 把根目录的 `index.html` 发给浏览器；而 `index.html` 里的 fetch 用的就是同源相对路径 `/api/optimize`、`/api/compare`，所以页面和接口在同一个 5000 端口上，无需 CORS、无需改 URL、无需另开静态服务器。

### 用的模型和端点

- **模型**：`glm-5.1`
- **端点**：`https://open.bigmodel.cn/api/anthropic`（litellm provider 前缀为 `anthropic`，走 Anthropic 消息格式）

> **为什么必须用 anthropic 端点而不是 openai/paas 端点**：GLM Coding 套餐**只在 anthropic 兼容端点计费**。如果改用 OpenAI 兼容的 `paas/v4` 端点，会直接返回 HTTP 429、错误码 `1113`（余额不足）。本项目的 GLM 接入已经按 anthropic 端点写死，开箱即用。

provider 选择规则在 [api/dspy_lab.py](api/dspy_lab.py) 的 `_select_provider()`：环境里有 `GLM_API_KEY` 就走 GLM-5.1，否则走 DeepSeek。

### 自动回退到 DeepSeek

如果环境里**没有** `GLM_API_KEY`、**只有** `DEEPSEEK_API_KEY`，启动会自动回退到 DeepSeek（模型 `deepseek-v4-flash`，端点 `https://api.deepseek.com`），`start.bat` 会打印 `LLM provider: DeepSeek` 提示当前用的不是 GLM。两个 key 一个都没有则启动报错。

### Vercel 部署不受影响

Vercel 上只配了 `DEEPSEEK_API_KEY`、没有 `GLM_API_KEY`，所以线上一如既往走 DeepSeek；新增的 `GET /` 路由在 Vercel 上会被静态 `index.html` 托管覆盖，对本仓库的线上行为零影响。

## 本地调试（Vercel CLI，DeepSeek 侧）

若想用 `vercel dev` 调试线上同款 DeepSeek 流程（而非 GLM-5.1），照旧设 `DEEPSEEK_API_KEY`：

```bash
vercel env pull .env      # 或手写 .env：DEEPSEEK_API_KEY=sk-...
vercel dev                # 打开 http://localhost:3000
```

## 端到端验证

**一键自动化**（需 key）：`test_e2e.py` 用 Flask test client 跑通 optimize → compare 全链路。
```bash
export DEEPSEEK_API_KEY=sk-xxx   # Windows git bash
PYTHONPATH=api uv run --no-project --with 'dspy==3.2.1' --with 'flask>=3.0,<4.0' python test_e2e.py
```

**手动 curl**：
```bash
# 存活探测（不花钱）。本地 start.bat 在 http://127.0.0.1:5000 ；vercel dev 在 http://localhost:3000
curl http://127.0.0.1:5000/api/health

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

- teacher、裁判、对比模型三者复用同一个被选中的 provider + key（GLM 或 DeepSeek）。GLM 现已是本地默认 provider：设 `GLM_API_KEY` 即自动启用智谱 GLM-5.1（anthropic 端点），无需改代码。
- 模型 / 端点均由 [api/dspy_lab.py](api/dspy_lab.py) 的 `_select_provider()` 从环境变量决定（导出常量为 `LLM_MODEL` / `LLM_BASE` / `LLM_PREFIX` / `KEY_ENV`）；切换模型改这里的逻辑即可。
