import { useEffect, useState } from 'react'
import { Activity, Wallet } from 'lucide-react'
import ExecutionPage from './ExecutionPage'
import { api } from './api'

const views = [{ id: 'execution', name: '执行层', icon: Wallet }]

type System = Record<string, any>

export default function App() {
  const [system, setSystem] = useState<System>({})
  const [error, setError] = useState('')
  const [mobile, setMobile] = useState(false)

  useEffect(() => {
    api<System>('/system')
      .then(setSystem)
      .catch((problem) => setError(String(problem?.message || problem)))
  }, [])

  return (
    <div className="app">
      <aside className={'sidebar ' + (mobile ? 'open' : '')}>
        <a className="brand" href="#execution">
          <span className="brand-mark">
            <Activity size={23} />
          </span>
          <span>
            A-SHARE <b>LAB</b>
            <small>执行层工作台</small>
          </span>
        </a>
        <div className="nav-caption">账户</div>
        <nav>
          {views.map((view) => (
            <a key={view.id} href={'#' + view.id} className="active">
              <view.icon size={18} />
              <span>{view.name}</span>
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="environment-label">LOCAL ENVIRONMENT</span>
          <strong>{system.environment || 'Python'}</strong>
          <small>{system.price_basis || 'hfq_research'}</small>
        </div>
      </aside>
      {mobile && <button className="scrim" aria-label="关闭导航" onClick={() => setMobile(false)} />}
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">执行层</div>
          <div className="connection">{error ? '连接失败' : '本地只读'}</div>
        </header>
        {error && <div className="error-banner">{error}</div>}
        <main className="content">
          <ExecutionPage fail={(problem) => setError(String((problem as Error)?.message || problem))} />
        </main>
      </div>
    </div>
  )
}
