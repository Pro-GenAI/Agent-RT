/**
 * Artie's moods: a shield pose when Guard blocks a call, a celebration when a migration passes.
 * ASCII only so legacy Windows consoles render them. Keep in step with python/src/ext/artie_moods.py.
 */
export const MOODS = {
	// Narrowed eyes, arm out, shield up.
	shield: [
		'   .--------.    ______ ',
		'  /.--------.\\  |  ||  |',
		'  || >    < ||==|==||==|',
		"  |'--------'|  \\  ||  /",
		"   '-.____.-'    '.__.' ",
	],
	// Arms up, confetti.
	celebrate: [
		" *  .  '  *  .  '  *  . ",
		'  \\   .--------.   /    ',
		'   \\ /.--------.\\ /     ',
		'    \\|| ^    ^ ||/      ',
		"     |'--------'|       ",
		"      '-.____.-'        ",
	],
} as const;

/** Artie in `mood` with `caption` beside the eyes. */
export function pose(mood: keyof typeof MOODS, caption: string): string {
	const lines: string[] = [...MOODS[mood]];
	const eyes = lines.findIndex(
		(line) => line.includes('^') || line.includes('>'),
	);
	lines[eyes] += `  ${caption}`;
	return lines.map((line) => line.trimEnd()).join('\n');
}
