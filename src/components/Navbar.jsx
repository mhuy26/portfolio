import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav style={{ padding: "1rem", textAlign: "center", background: "#1e1e1e" }}>
      <Link to="/" style={{ margin: "1rem", color: "#e0e0e0", textDecoration: "none" }}>Home</Link>
      <Link to="/projects" style={{ margin: "1rem", color: "#e0e0e0", textDecoration: "none" }}>Projects</Link>
      <Link to="/contact" style={{ margin: "1rem", color: "#e0e0e0", textDecoration: "none" }}>Contact</Link>
    </nav>
  );
}
