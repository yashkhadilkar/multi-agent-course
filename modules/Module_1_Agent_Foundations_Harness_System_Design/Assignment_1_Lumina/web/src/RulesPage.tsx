import rule from './rule.json';

/**
 * /rules — the proposed bonus rule. The content lives in rule.json so it can be updated
 * without touching this page.
 */
export function RulesPage() {
  return (
    <>
      <header className="topbar">
        <div className="brand">
          LUM<span>INA</span>
        </div>
        <nav>
          <a href="/">Ask</a>
          <a href="/evals">Evals</a>
          <a href="/rules" className="on">
            Rules
          </a>
        </nav>
      </header>
      <div className="evals">
        <h1>
          <code>{rule.id}</code> · {rule.title}
        </h1>
        <div className="sub">
          <span className="pill">{rule.type}</span>{' '}
          <span className={`pill ${rule.severity === 'error' ? 'error' : 'cap'}`}>{rule.severity}</span>
        </div>

        <dl className="design">
          <dt>Rule</dt>
          <dd>{rule.rule}</dd>
          <dt>Precedent</dt>
          {rule.precedent.map((p, i) => (
            <dd key={i}>{p}</dd>
          ))}
          <dt>Self-check</dt>
          <dd>{rule.selfCheck}</dd>
        </dl>
      </div>
    </>
  );
}
