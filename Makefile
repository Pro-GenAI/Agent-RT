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
# Positional syntax: make eval-swarmguard 3  (three independent trials per case).
# Make treats "3" as a second goal; declare only that numeric goal as a no-op.
ifeq ($(firstword $(MAKECMDGOALS)),eval-swarmguard)
SWARMGUARD_REPEATS := $(or $(word 2,$(MAKECMDGOALS)),1)
ifneq ($(word 2,$(MAKECMDGOALS)),)
ifneq ($(word 3,$(MAKECMDGOALS)),)
$(error eval-swarmguard takes only one positional repeat count)
endif
.PHONY: $(word 2,$(MAKECMDGOALS))
$(word 2,$(MAKECMDGOALS)):
	@:
endif
else
SWARMGUARD_REPEATS := 1
endif
.PHONY: eval-swarmguard

eval-swarmguard:
	@printf '[SwarmGuard] Initializing evaluation...\n' >&2
	@python -u benchmarks/swarmguard/run.py --config ./agent-rt.yaml --repeats "$(SWARMGUARD_REPEATS)" $(if $(strip $(SUBAGENT_MODEL)),--subagent-model "$(SUBAGENT_MODEL)")
