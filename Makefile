.PHONY: verify python-verify ts-verify

# Runs both implementations' full gate; make stops at the first failing step.
verify: python-verify ts-verify

python-verify:
	$(MAKE) -C python verify

ts-verify:
	$(MAKE) -C typescript verify
