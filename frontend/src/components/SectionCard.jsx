export default function SectionCard({ title, subtitle, children, action, className = "" }) {
  return (
    <section className={`panel-card section-card ${className}`.trim()}>
      <div className="section-head">
        <div>
          <h2 className="section-title">{title}</h2>
          {subtitle ? <p className="section-subtitle">{subtitle}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
