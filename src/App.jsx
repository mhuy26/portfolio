import Navbar from "./components/Navbar";
import Home from "./components/Home";
import Projects from "./components/Projects";
import Contact from "./components/Contact";
import { Routes, Route } from "react-router-dom";  // ✅ Import Routes and Route


// Import Project Pages
import LoanRepayment from "./components/projects/LoanRepayment";
import SMSDetection from "./components/projects/SMSDetection";
import RandomForest from "./components/projects/RandomForest";

export default function App() {
  return (
    <>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/contact" element={<Contact />} />
        
        {/* Project Pages */}
        <Route path="/loan-repayment" element={<LoanRepayment />} />
        <Route path="/sms-detection" element={<SMSDetection />} />
        <Route path="/random-forest" element={<RandomForest />} />
      </Routes>
    </>
  );
}
