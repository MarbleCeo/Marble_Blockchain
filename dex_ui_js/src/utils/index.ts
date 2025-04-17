export * from './safe-math'
export * from './tokens'
export * from './pools'
export * from './ids'
export * from './layouts'
export * from './web3'

export function getUnixTs(): number {
  return new Date().getTime() / 1000
}

export function parseError(err: any): string {
  if (err instanceof Error) {
    return err.message
  }
  return String(err)
}

// eslint-disable-next-line
export async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export function getUnixTs() {
  return new Date().getTime()
}

export function parseError(error: any) {
  const { response } = error
  let status, code, message

  if (response) {
    // request sent
    status = response.data.status || response.status
    code = response.data.code || response.statusText
    message = response.data.message || error.message
  } else {
    // request did not sent
    status = 504
    code = 'Request timeout'
    message = 'Please try again later'
  }

  return {
    status,
    code,
    message
  }
}
