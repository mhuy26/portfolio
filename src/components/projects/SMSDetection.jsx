export default function SMSDetection() {
    return (
      <div style={{ textAlign: "center", padding: "2rem" }}>
        <h2>SMS Spam Detection - NLP</h2>
        <img src="/projects/sms-spam.png" alt="Spam Detection" style={{ width: "70%", borderRadius: "8px" }} />
        <p>
          Built a spam SMS detection system using **Scikit-learn, Pandas, and Numpy**, trained on 5,000+ SMS messages.
        </p>
        <p>
          The model achieved **96% accuracy** using text pre-processing (punctuation removal, stop words, tokenization, 
          Bag of Words, and TF-IDF) and a **Naive Bayes classifier**.
        </p>
        <a href="https://github.com/mhuy26/sms-spam-detection" target="_blank" rel="noopener noreferrer"
          style={{ color: "#61dafb", textDecoration: "none", fontWeight: "bold" }}>
          View on GitHub →
        </a>
      </div>
    );
  }
  