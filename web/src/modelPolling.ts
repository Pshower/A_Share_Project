type Row = Record<string, any>
export const terminal = new Set(['succeeded', 'failed', 'cancelled', 'interrupted', 'blocked'])

export async function pollModels(runs: Row[], fetchJson: (path: string) => Promise<Row>): Promise<Row[]> {
  return Promise.all(runs.filter((r) => r.job_id && !terminal.has(r.status)).map(async (run) => {
    try {
      const job = await fetchJson('/jobs/' + run.job_id)
      const plan = job.status === 'succeeded' && job.result?.snapshot_id
        ? await fetchJson('/market/snapshots/plans/' + job.result.snapshot_id) : null
      return { ...run, status: job.status, errors: job.error ? [job.error] : [], plan, read_error: null }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return { ...run, read_error: message, errors: [message] }
    }
  }))
}
