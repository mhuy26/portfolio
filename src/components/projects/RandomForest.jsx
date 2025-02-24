export default function RandomForest() {
    return (
      <div style={{ textAlign: "center", padding: "2rem" }}>
        <h2>Decision Tree & Random Forest Loan Prediction</h2>
        <img src="/projects/random-forest.png" alt="Random Forest Model" style={{ width: "70%", borderRadius: "8px" }} />
        <p>
          Predicted loan repayment using **Decision Trees and Random Forests**, achieving **85% accuracy**.
        </p>
        <p>
          Tuned hyperparameters with **GridSearchCV**, handling missing values and preparing features for
          **10,000+ loan records**.
        </p>
        <a href="https://github.com/mhuy26/loan-repayment-ml" target="_blank" rel="noopener noreferrer"
          style={{ color: "#61dafb", textDecoration: "none", fontWeight: "bold" }}>
          View on GitHub →
        </a>
      </div>
    );
  }
  