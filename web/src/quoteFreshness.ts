export const QUOTE_TTL_MS = 120_000

export function freshQuote<T extends { id?: string; status?: string; created_at?: string }>(
  quotes: T[],
  now = Date.now()
) {
  const cutoff = now - QUOTE_TTL_MS
  return quotes.find((quote) => {
    const created = Date.parse(quote.created_at || '')
    return quote.status === 'ready' && Number.isFinite(created) && created >= cutoff
  })
}
