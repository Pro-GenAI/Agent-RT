// ESM entry for `agent-rt/anthropic` and `agent-rt/@anthropic-ai/sdk`: the
// package builds as CommonJS, where an ES module's default import would be the
// whole exports object instead of the client class. Re-exporting keeps a
// single shared module instance.
import { Anthropic } from './anthropic.js';

export * from './anthropic.js';
export default Anthropic;
