import {
  Activity,
  AlertTriangle,
  Gauge,
  Home,
  ShieldCheck,
} from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "System Overview", icon: Home },
  { to: "/drift-monitoring", label: "Drift Monitoring", icon: AlertTriangle },
  { to: "/rate-control", label: "Rate Control", icon: Gauge },
  { to: "/self-healing", label: "Model Governance", icon: ShieldCheck },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-icon">
          <Activity size={20} />
        </span>
        <div>
          <p className="sidebar-kicker">Industrial IoT</p>
          <h1 className="sidebar-title">Self-Healing ML Operations</h1>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? "is-active" : ""}`
              }
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}
