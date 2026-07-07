// Groups a flat word stream into "lines" for display modes that need
// sentence-scale chunks (Music Video, Teleprompter) rather than individual
// words. There are no timestamps to key off (deliberately dropped for
// speed — see useLiveTranscription), so this is a text-only heuristic:
// break on sentence-ending punctuation when present (Whisper's .en models
// do emit it), otherwise fall back to a max-words-per-line cap so a line
// never grows unbounded on unpunctuated speech.
export function groupWordsIntoLines(words, { maxWordsPerLine = 9 } = {}) {
  const lines = [];
  let current = [];

  for (const w of words) {
    current.push(w);
    const endsSentence = /[.!?]$/.test(w.text);
    if (endsSentence || current.length >= maxWordsPerLine) {
      lines.push({ id: current[0].id, words: current });
      current = [];
    }
  }

  // The trailing in-progress line has no closing punctuation yet, but it's
  // still real speech that's already been transcribed — keep it as the
  // current "open" line instead of dropping it until it happens to close.
  if (current.length) {
    lines.push({ id: current[0].id, words: current });
  }

  return lines;
}
