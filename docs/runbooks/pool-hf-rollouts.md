# Pool + HF Rollouts Runbook

## Pool Runtime Findings

- pool version: `1.0.4`
- config directory: `/root/.config/poolside`
- log directory: `/root/.local/state/poolside/logs`
- trajectory directory: `/root/.local/state/poolside/trajectories`
- credentials status: `/root/.config/poolside/credentials.json` not present on the B300 VM

## Install

Headless install requires accepting the EULA explicitly:

```bash
POOL_INSTALL_ACCEPT_EULA=1 POOL_INSTALL_UPDATE_PATH=0 POOL_INSTALL_SPLASH=0 \
  curl -fsSL https://downloads.poolside.ai/pool/install.sh | \
  POOL_INSTALL_ACCEPT_EULA=1 POOL_INSTALL_UPDATE_PATH=0 POOL_INSTALL_SPLASH=0 sh
```

Then:

```bash
export PATH="$HOME/.local/bin:$PATH"
pool --version
pool config
```

## Dummy ACP Spike

Run pool against the project-owned dummy ACP server:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /root/hackathon/laguna-xs2-expert-coactivation-scheduling/.worktrees/pool-hf-acp-spike
pool --agent-server "uv run python scripts/run_hf_laguna_acp_server.py --mode dummy" \
  exec -p "Reply READY" -o json --unsafe-auto-allow
```

Incoming and outgoing ACP messages are logged to:

```text
runs/acp_spike/messages.jsonl
```

Default Poolside backend is for scaffold inspection only. Activation-producing rollouts must use the HF ACP server.
