/**
 * HTTP Utilities - Better error handling for API calls
 */

export async function fetchJson(url: string) {
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } })
  const txt = await res.text()
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`)
  }
  try {
    return JSON.parse(txt)
  } catch {
    throw new Error(`Bad JSON from ${url}: ${txt.slice(0, 200)}`)
  }
}
