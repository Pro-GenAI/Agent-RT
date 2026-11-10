// ESM entry for `agent-rt/openai`: the package builds as CommonJS, where an ES
// module's default import would be the whole exports object instead of the
// client class. Re-exporting keeps a single shared module instance.
import { OpenAI } from './openai.js';

export * from './openai.js';
export default OpenAI;
