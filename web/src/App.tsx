import { useEffect, useState } from 'react'
import { Activity, Wallet } from 'lucide-react'
import ExecutionPage from './ExecutionPage'
import { api } from './api'

const sections = [
  { id: 'books', name: '三本账' },
  { id: 'recommend', name: '每日推荐' },
  { id: 'shadow', name: '影子记录' },
]

type System = Record<string, any>

function parseRoute() {
  const raw = location.hash.replace(/^#/, '')
  const [view, section] = raw.split('/')
  const known = sections.some((item) => item.id === section)
  return { view: view || 'execution', section: known ? section : 'books' }
}

export default function App() {
  const [system, setSystem] = useState<System>({})
  const [error, setError] = useState('')
  const [mobile, setMobile] = useState(false)
  const [route, setRoute] = useState(parseRoute)

  useEffect(() => {
    api<System>('/system')
      .then(setSystem)
      .catch((problem) => setError(String(problem?.message || problem)))
  }, [])

  useEffect(() => {
    const change = () => {
      setRoute(parseRoute())
      setMobile(false)
    }
    addEventListener('hashchange', change)
    return () => removeEventListener('hashchange', change)
  }, [])

  const current = sections.find((item) => item.id === route.section) || sections[0]

  return (
    <div className="app">
      <aside className={'sidebar ' + (mobile ? 'open' : '')}>
        <a className="brand" href="#execution/books">
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
          <span className="nav-parent">
            <Wallet size={18} />
            <span>执行层</span>
          </span>
          <div className="nav-sub">
            {sections.map((item) => (
              <a
                key={item.id}
                href={'#execution/' + item.id}
                className={route.section === item.id ? 'active' : ''}
                aria-current={route.section === item.id ? 'page' : undefined}
              >
                {item.name}
              </a>
            ))}
          </div>
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
          <div className="breadcrumb">
            执行层 <span>/</span> {current.name}
          </div>
          <div className="connection">{error ? '连接失败' : '本地只读'}</div>
        </header>
        {error && <div className="error-banner">{error}</div>}
        <main className="content">
          <ExecutionPage
            key={route.section}
            section={route.section}
            fail={(problem) => setError(String((problem as Error)?.message || problem))}
          />
        </main>
      </div>
    </div>
  )
}
