.PHONY: verify python-verify ts-verify

# Runs both implementations' full gate; make stops at the first failing step.
verify: python-verify ts-verify

python-verify:
	$(MAKE) -C python verify

ts-verify:
	$(MAKE) -C typescript verify

# Strict live comparison using the repository's explicitly selected YAML.
# Both workers and the mandatory Decision endpoint are configured there.
# No EVAL_* overrides or separate benchmark modes.
# Optional SUBAGENT_MODEL=<model_list alias> selects one worker model for
# all cases in both harnesses; the reviewer/Decision models stay configured.
# eval-swarmguard-alt is a shortcut for the registered subagent-alt alias.
.PHONY: eval-swarmguard eval-swarmguard-alt

eval-swarmguard:
	@printf '[SwarmGuard] Initializing evaluation...\n' >&2
	@python -u benchmarks/swarmguard/run.py --config ./agent-rt.yaml $(if $(strip $(SUBAGENT_MODEL)),--subagent-model "$(SUBAGENT_MODEL)")

eval-swarmguard-alt:
	@$(MAKE) --no-print-directory eval-swarmguard SUBAGENT_MODEL=subagent-alt
