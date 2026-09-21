"""HumanEval harness: generate k completions per problem from any OpenAI-compatible endpoint, execute
each against the problem's real tests in an isolated container, score pass@k.

Reuses from bench: `harness.python_adapter.PythonAdapter.parse_results` (pytest JSON report -> {test_id: bool}),
`PythonCollectionFailure` semantics, the Sandbox `run(cmd, cwd, timeout, extra_env)` interface, and
`pipeline.stats` (unbiased pass@k + bootstrap). It deliberately does NOT reuse PythonAdapter.install/run_tests:
those build a per-repo host venv, which cannot be mounted into a network-less container; the runner image
(Dockerfile here) carries pinned pytest instead. No agent loop, no tool calls.
"""
