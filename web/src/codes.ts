export function parseCodes(text: string) {
  return text
    .split(/[\s,，;；]+/)
    .map((value) => value.trim())
    .filter(Boolean)
}
