// Keep only ONNX Runtime's native binaries for the image's OS and architecture.
// Run in the same Docker layer as npm install: deleted files then add no image size.
// Other packages are deliberately untouched.
import { existsSync, readdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';

const modules = resolve(process.argv[2] ?? '/app/node_modules');
const bin = join(modules, 'onnxruntime-node', 'bin');
if (!existsSync(bin)) {
	throw new Error(`Missing onnxruntime-node binaries: ${bin}`);
}

const subdirs = (path) =>
	readdirSync(path, { withFileTypes: true }).filter((entry) =>
		entry.isDirectory(),
	);
let removed = 0;
let kept = 0;
for (const napi of subdirs(bin)) {
	const napiPath = join(bin, napi.name);
	for (const os of subdirs(napiPath)) {
		const osPath = join(napiPath, os.name);
		if (os.name !== process.platform) {
			rmSync(osPath, { recursive: true, force: true });
			removed++;
			continue;
		}
		for (const arch of subdirs(osPath)) {
			if (arch.name === process.arch) {
				kept++;
			} else {
				rmSync(join(osPath, arch.name), {
					recursive: true,
					force: true,
				});
				removed++;
			}
		}
	}
}
if (kept === 0) {
	throw new Error(
		`No ONNX Runtime binaries for ${process.platform}/${process.arch} in ${bin}`,
	);
}
console.log(
	`Kept ONNX Runtime ${process.platform}/${process.arch}; pruned ${removed} other platform directories.`,
);
