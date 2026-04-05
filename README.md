# Multi-Agent Opinion Simulator

Watch AI agents with distinct personalities debate any news article or topic in real time — and jump in yourself mid-debate to steer the conversation.

## Quick Start

```bash
git clone <repo-url> && cd Simulation-hack
pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY inside .env
uvicorn main:app --reload
# open http://localhost:8000
```

## How It Works

1. **Configure** — Paste an article or topic, pick a Gemini model, set the number of agents and rounds. Customize each agent's personality or use the 5 defaults (Skeptic, Idealist, Pragmatist, Traditionalist, Analyst). The live cost estimator shows total API calls and estimated spend before you run.

2. **Watch** — Agents debate round by round, each responding to what others said. New posts animate in as they arrive (polling every 2 seconds).

3. **Join in** — While the simulation is running, type a question or comment in the "Join the Discussion" box. Your message is injected into the shared feed and agents will respond to it in subsequent calls.

4. **Summary** — When complete, a summary shows each agent's post count, word count, and how their tone evolved from first to last post.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Google AI Studio API key |
| `ALLOWED_ORIGIN` | `http://localhost:8000` | CORS allowed origin (set to your domain in prod) |
| `TRUST_PROXY_IP` | `false` | Set `true` if behind a reverse proxy (reads `X-Forwarded-For`) |

## Security Notes

- The Gemini API key is **never** sent to the frontend — all AI calls happen server-side.
- Per-IP rate limit: 5 simulation starts per 10 minutes.
- Article max 5,000 characters; agent system prompts max 500 characters.
- Prompt injection patterns are detected and rejected server-side.
- `/status` and `/inject` endpoints are IP-gated — only the client that started a simulation can read or interact with it.
- Full article content is never logged; only simulation metadata is recorded.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List available Gemini models with pricing |
| `POST` | `/simulate` | Start a simulation, returns `simulation_id` immediately |
| `GET` | `/status/{id}` | Poll simulation status and feed (IP-gated) |
| `POST` | `/inject/{id}` | Inject a user message into a running simulation (IP-gated) |
