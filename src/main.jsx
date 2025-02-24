import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";  // ✅ BrowserRouter is only here
import App from "./App.jsx";
import "./App.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter basename="/portfolio">  {/* ✅ Correct place for BrowserRouter */}
      <App />
    </BrowserRouter>
  </StrictMode>
);
