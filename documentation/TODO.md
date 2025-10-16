# TODO

# P0
- [ ] OpenWebUI integration needs work. Particularly I belive it is only the validation and health check that needs to become consistent with Ollama 
- [x] hugging-llama catalog --memory 24GB does not pull all of the models  
- [x] There is a problem with the `pull` command on Windows. It shows `pulling manifest` but it does not appear to be working on Win 11. **Solution** `curl` is a bit clunky on PowerShell, ask the users to just use WSL on Windows to use consisten commands with Mac/Linux but benefit from better GPU 
      
# P2
- [ ] Add analytics pipeline for request tracking and latency dashboards.
- [ ] Implement improved parallel serving to maximize hardware utilization across models.
- [ ] Support model warm pools and configurable eviction strategies.
- [ ] Add autoscaling hooks for dynamic worker provisioning.
- [ ] Expose health and readiness probes with structured metrics.
- [ ] Provide pluggable authentication and rate limiting middleware.
- [ ] Build UI for monitoring active sessions and throughput trends.

## Port Configuration Instructions

- Set `OLLAMA_LOCAL_PORT` to change the default API port used by the CLI. The `serve`
  command will use it unless `--port` is explicitly supplied, and other subcommands
  will target the matching URL.
