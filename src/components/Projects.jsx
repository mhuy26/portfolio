import { useNavigate } from "react-router-dom";

export default function Projects() {
  const navigate = useNavigate();

  return (
    <div style={{ textAlign: "center", padding: "2rem" }}>
      <h2>My Projects</h2>
      <div style={{
        display: "flex",
        flexDirection: "column",
        gap: "15px",
        maxWidth: "400px",
        margin: "auto",
      }}>
        <button onClick={() => navigate("/loan-repayment")} style={buttonStyle}>Loan Repayment (Keras, TensorFlow)</button>
        <button onClick={() => navigate("/sms-detection")} style={buttonStyle}>SMS Spam Detection (NLP)</button>
        <button onClick={() => navigate("/random-forest")} style={buttonStyle}>Random Forest Loan Prediction</button>
      </div>
    </div>
  );
}

const buttonStyle = {
  background: "#1e1e1e",
  color: "#e0e0e0",
  border: "none",
  padding: "10px",
  fontSize: "16px",
  cursor: "pointer",
  borderRadius: "5px",
  transition: "0.3s",
  boxShadow: "0px 0px 10px rgba(255, 255, 255, 0.2)"
};
