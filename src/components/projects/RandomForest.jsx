export default function RandomForest() {
  const isMobile = window.innerWidth <= 768;
  const headingStyle = {
    fontSize: isMobile ? "1.5rem" : "2rem",
    fontWeight: "bold",
    marginBottom: "1rem",
  };
  const listStyle = {
    listStyleType: "disc",
    paddingLeft: "20px",
    fontSize: isMobile ? "1rem" : "1.1rem",
  };
  const paragraphStyle = {
    fontSize: isMobile ? "1rem" : "1.1rem",
    marginBottom: "1rem",
  };
  const subheadingStyle = {
    fontSize: isMobile ? "1.4rem" : "1.5rem",
    fontWeight: "bold",
    marginBottom: "1rem",
  };

  return (
    <div style={{ textAlign: "center", padding: "2rem", maxWidth: "95%", margin: "auto" }}>
      <h2 style={headingStyle}>
        🌳 Decision Tree & Random Forest Loan Prediction
      </h2>

      {/* Project Summary */}
      <div style={{ textAlign: "left", maxWidth: "80%", margin: "auto", paddingTop: "1rem" }}>
        <ul style={listStyle}>
          <li>Developed a loan repayment prediction model using <strong>Decision Trees</strong> and <strong>Random Forests</strong>.</li>
          <li>Achieved an <strong>85% accuracy</strong> by tuning hyperparameters with <strong>GridSearchCV</strong>.</li>
          <li>Processed and cleaned <strong>10,000+ loan records</strong>, handling missing values and feature engineering.</li>
        </ul>
      </div>

      {/* EDA Section */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "90%", margin: "auto" }}>
        {/* FICO Score Analysis */}
        <p style={paragraphStyle}>
          To understand loan approval trends, we analyzed the <strong>FICO score distribution</strong> for different credit policies.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", gap: "3rem" }}>
          {/* Left: Histogram */}
          <img 
            src="/portfolio/img_RF-DTree/hist_FICOScore_MeetCreditPolicy.png"  
            alt="FICO Score Histogram"
            style={{ width: "100%", borderRadius: "10px" }}
          />

          {/* Right: Insights - Added more padding-left */}
          <div style={{ paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📊 FICO Score & Credit Policy</h3>
            <ul style={listStyle}>
              <li>Higher <strong>FICO scores</strong> increase the chances of loan approval.</li>
              <li>Loans with lenient policies allow <strong>lower FICO scores</strong>, increasing default risk.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Interest Rate vs FICO Score */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "90%", margin: "auto" }}>
        <p style={paragraphStyle}>
          The relationship between <strong>interest rates</strong> and <strong>FICO scores</strong> shows a clear inverse trend.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", gap: "3rem" }}>
          {/* Left: Graph */}
          <img 
            src="/portfolio/img_RF-DTree/joint_FICOvsInterestRate.png"  
            alt="Interest Rate vs FICO Score"
            style={{ width: "100%", borderRadius: "10px" }}
          />

          {/* Right: Insights - Added more padding-left */}
          <div style={{ paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📉 Interest Rate & FICO Score</h3>
            <ul style={listStyle}>
              <li>Higher <strong>FICO scores</strong> result in <strong>lower interest rates</strong>.</li>
              <li>Lower scores lead to <strong>higher interest rates</strong> due to increased risk.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Loan Purpose Analysis */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "90%", margin: "auto" }}>
        <p style={paragraphStyle}>
          Different loan purposes have varying levels of risk, as shown in the countplot below.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", gap: "3rem" }}>
          {/* Left: Graph */}
          <img 
            src="/portfolio/img_RF-DTree/dist_paymentStatus_purpose.png"  
            alt="Loan Purpose Countplot"
            style={{ width: "100%", borderRadius: "10px" }}
          />

          {/* Right: Insights - Added more padding-left */}
          <div style={{ paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📋 Loan Purpose Analysis</h3>
            <ul style={listStyle}>
              <li><strong>Debt consolidation</strong> is the most common reason for loans.</li>
              <li>Loans for <strong>small businesses</strong> have a higher default rate.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Data Preprocessing */}
      <div style={{ textAlign: "center", padding: "2rem", maxWidth: "95%", margin: "auto" }}>
        <h2 style={headingStyle}>🔄 Data Preprocessing: Handling Categorical Features</h2>

        <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr", 
            alignItems: "center", 
            gap: "3rem",
            marginBottom: "2rem"
        }}>
          {/* Left: Code Block */}
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "10px",
            textAlign: "left",
            fontSize: "1.2rem",
            width: "100%",
            overflowX: "auto"
          }}>
            {`# Identify Categorical Feature
categorical_feats = ['purpose']

# Convert categorical feature into numerical using one-hot encoding
final_df = pd.get_dummies(data=df, columns=categorical_feats, dtype=int, drop_first=True)

# Display the first few rows of the transformed dataset
final_df.head()`}
          </pre>

          {/* Right: Explanation - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📌 One-Hot Encoding Categorical Variables</h3>
            <ul style={listStyle}>
              <li>The <strong>purpose</strong> column is categorical and must be converted to numeric.</li>
              <li>We use <strong>one-hot encoding</strong> to transform categories into multiple binary features.</li>
              <li><strong>pd.get_dummies()</strong> creates separate columns for each unique category.</li>
              <li>Setting <strong>drop_first=True</strong> prevents multicollinearity.</li>
            </ul>
          </div>
        </div>

        <p style={paragraphStyle}>
          Why? 🤔  
          Machine learning models cannot process categorical text data directly.  
          One-hot encoding transforms categories into numerical values while preserving their meaning,  
          allowing models to interpret and learn from categorical features effectively.
        </p>
      </div>

      {/* Training a Decision Tree Model */}
      <div style={{ textAlign: "center", padding: "2rem", maxWidth: "95%", margin: "auto" }}>
        <h2 style={headingStyle}>🌳 Training a Decision Tree Model</h2>

        {/* Grid for Code & Explanation */}
        <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr",
            alignItems: "center", 
            gap: "3rem", 
            marginBottom: "2rem"
        }}>
          {/* Left: Code Block */}
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "8px",
            textAlign: "left",
            overflowX: "auto",
            fontSize: "1.1rem",
            width: "100%"
          }}>
            {`from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model instance
dtree = DecisionTreeClassifier()

# Train the Decision Tree model
dtree.fit(X_train, y_train)`}
          </pre>

          {/* Right: Explanation Block - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📌 Decision Tree Classifier</h3>
            <ul style={listStyle}>
              <li>We use <strong>DecisionTreeClassifier()</strong> from <code>sklearn.tree</code> to build our model.</li>
              <li>The model is trained on the <strong>training dataset (X_train, y_train)</strong>.</li>
              <li>Decision trees are useful for capturing non-linear relationships in data.</li>
            </ul>
          </div>
        </div>

        {/* Transition to Predictions & Evaluation */}
        <p style={paragraphStyle}>
          Now that the Decision Tree model is trained, we evaluate its performance on the test dataset.
        </p>
      </div>

      {/* Decision Tree Evaluation */}
      <div style={{ textAlign: "center", padding: "2rem", maxWidth: "90%", margin: "auto" }}>
        <h3 style={subheadingStyle}>
          📋 Decision Tree Evaluation
        </h3>

        {/* Grid Layout for Classification Report */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", gap: "3rem", marginBottom: "2rem" }}>
          
          {/* Left: Classification Report Image */}
          <img 
            src="/portfolio/img_RF-DTree/dtree_evaluation.png"  
            alt="Decision Tree Classification Report"
            style={{ width: "100%", borderRadius: "8px" }}
          />

          {/* Right: Explanation - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>�� Key Insights</h3>
            <ul style={listStyle}>
              <li><strong>Overall Accuracy:</strong> The Decision Tree model achieves <strong>73% accuracy</strong>.</li>
              <li><strong>Class 0 (Fully Paid Loans):</strong> Performs well with <strong>85% precision and 82% recall</strong>.</li>
              <li><strong>Class 1 (Not Fully Paid Loans):</strong> Recall remains <strong>low at 23%</strong>, indicating difficulty detecting loan defaults.</li>
              <li>This suggests <strong>class imbalance is affecting the model's ability to detect minority class samples.</strong></li>
            </ul>
          </div>
        </div>

        {/* Grid Layout for Confusion Matrix */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", gap: "3rem", marginBottom: "2rem" }}>
          
          {/* Left: Confusion Matrix Image */}
          <img 
            src="/portfolio/img_RF-DTree/dtree_confussionMatrix.png"  
            alt="Decision Tree Confusion Matrix"
            style={{ width: "100%", borderRadius: "8px" }}
          />

          {/* Right: Explanation - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📊 Confusion Matrix Insights</h3>
            <ul style={listStyle}>
              <li>The model <strong>correctly classified 1,995 fully paid loans</strong> but misclassified <strong>436 as defaults</strong>.</li>
              <li>For <strong>Not Fully Paid Loans</strong>, it correctly identified <strong>100</strong> but misclassified <strong>343</strong>, highlighting its struggles with recall.</li>
              <li><strong>High false negatives</strong> indicate that many actual defaulters were predicted as fully paid, which is problematic for loan risk assessment.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Training the Random Forest Model */}
      <div style={{ textAlign: "center", padding: "2rem", maxWidth: "95%", margin: "auto" }}>
        <h2 style={headingStyle}>🌲 Training the Random Forest Model</h2>

        {/* Code and Explanation Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "center", gap: "3rem", marginBottom: "2rem" }}>
          {/* Left: Code Block */}
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "8px",
            textAlign: "left",
            overflowX: "auto",
            fontSize: "1.1rem",
            width: "100%"
          }}>
            {`from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier with 600 trees
rfc = RandomForestClassifier(n_estimators=600)

# Train the model
rfc.fit(X_train, y_train)`}
          </pre>

          {/* Right: Explanation - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>🌟 Why Random Forest?</h3>
            <ul style={listStyle}>
              <li>Random Forest is an <strong>ensemble learning method</strong> that builds multiple decision trees and averages their predictions.</li>
              <li>Using <strong>600 trees</strong> helps improve the <strong>model's accuracy and robustness</strong> against overfitting.</li>
              <li>The model is trained using <strong>X_train</strong> and <strong>y_train</strong>, learning patterns in loan repayment behavior.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Random Forest Evaluation */}
      <div style={{ textAlign: "center", padding: "3rem", maxWidth: "90%", margin: "auto" }}>
        <h2 style={headingStyle}>
          🌲 Random Forest Evaluation
        </h2>

        {/* Grid Layout for Classification Report & Insights */}
        <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr",
            alignItems: "center", 
            gap: "3rem",
            marginBottom: "2rem"
        }}>
          {/* Left: Classification Report */}
          <img 
            src="/portfolio/img_RF-DTree/randForest_evaluation.png"  
            alt="Classification Report: Random Forest"
            style={{ width: "100%", borderRadius: "10px" }}
          />

          {/* Right: Key Insights - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>📌 Key Insights</h3>
            <ul style={listStyle}>
              <li><strong>Overall Accuracy:</strong> The model achieves <strong>85% accuracy</strong>, improving over Decision Tree.</li>
              <li><strong>Class 0 (Fully Paid Loans):</strong> Excellent recall (<strong>100%</strong>), meaning all fully paid loans were correctly classified.</li>
              <li><strong>Class 1 (Not Fully Paid Loans):</strong> Recall is <strong>only 3%</strong>, showing that the model still struggles to detect loan defaults.</li>
              <li>Despite higher accuracy, <strong>class imbalance remains a major issue</strong>, affecting the recall for minority classes.</li>
            </ul>
          </div>
        </div>

        {/* Confusion Matrix Section */}
        <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr",
            alignItems: "center", 
            gap: "3rem",
            marginTop: "2rem"
        }}>
          {/* Left: Confusion Matrix Image */}
          <img 
            src="/portfolio/img_RF-DTree/randForest_confusionMatrix.png"  
            alt="Confusion Matrix: Random Forest"
            style={{ width: "100%", borderRadius: "10px" }}
          />

          {/* Right: Explanation of Confusion Matrix - Added more padding-left */}
          <div style={{ textAlign: "left", paddingLeft: "2rem" }}>
            <h3 style={subheadingStyle}>🔍 Confusion Matrix Analysis</h3>
            <ul style={listStyle}>
              <li>The model correctly classifies most <strong>fully paid loans (1995 out of 2431)</strong>.</li>
              <li>However, it struggles with <strong>not fully paid loans</strong>, misclassifying <strong>343 out of 443</strong>.</li>
              <li>This imbalance suggests the model is <strong>biased toward predicting loans as fully paid</strong>.</li>
              <li>Further improvements, such as <strong>class weighting or oversampling</strong>, may help in detecting defaults better.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Model Comparison Section */}
      <div style={{ textAlign: "center", padding: "3rem", maxWidth: "90%", margin: "auto" }}>
        <h2 style={headingStyle}>
          🔍 Decision Tree vs. Random Forest: Performance Comparison
        </h2>

        {/* Centered Model Comparison Table */}
        <img 
          src="/portfolio/img_RF-DTree/model_compare.png"  
          alt="Model Comparison Table"
          style={{ width: "80%", borderRadius: "10px", marginBottom: "2rem" }}
        />

        {/* Key Observations */}
        <div style={{ textAlign: "left", maxWidth: "80%", margin: "auto", fontSize: "1.4rem" }}>
          <h3 style={subheadingStyle}>📌 Key Observations</h3>
          <ul style={listStyle}>
            <li><strong>Random Forest achieves 85% accuracy</strong>, outperforming Decision Tree's 73%.</li>
            <li><strong>Fully Paid Loans (Class 0)</strong> are well classified in both models, but Random Forest reaches 100% recall.</li>
            <li><strong>Loan Defaults (Class 1) are poorly detected</strong>: Decision Tree has 23% recall, but Random Forest drops to 3% recall.</li>
            <li>Random Forest shows stronger bias towards predicting loans as fully paid, reducing its effectiveness in risk assessment.</li>
          </ul>
        </div>

        {/* Advantages & Disadvantages Section */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", alignItems: "start", gap: "3rem", marginBottom: "2rem" }}>
          
          {/* Left: Advantages */}
          <div style={{ textAlign: "left" }}>
            <h4 style={subheadingStyle}>✅ Advantages of RF model</h4>
            <ul style={listStyle}>
              <li><strong>Higher Accuracy:</strong> Achieves 85% accuracy, outperforming Decision Tree.</li>
              <li><strong>Less Overfitting:</strong> Uses multiple trees to improve generalization performance.</li>
              <li><strong>Handles Missing Data:</strong> Can work well even with incomplete datasets.</li>
              <li><strong>More Robust:</strong> Less sensitive to outliers and noise.</li>
            </ul>
          </div>

          {/* Right: Disadvantages */}
          <div style={{ textAlign: "left" }}>
            <h4 style={subheadingStyle}>🚨 Disadvantages of RF model</h4>
            <ul style={listStyle}>
              <li><strong>Poor Recall for Class 1:</strong> Struggles to detect loan defaults (Class 1).</li>
              <li><strong>Computationally Expensive:</strong> Training with 600 trees increases runtime.</li>
              <li><strong>Less Interpretability:</strong> Unlike Decision Trees, it's harder to extract rules.</li>
              <li><strong>High Memory Usage:</strong> Requires more RAM due to multiple decision trees.</li>
            </ul>
          </div>
        </div>

        {/* Recommendations for Improvement */}
        <div style={{ textAlign: "left", maxWidth: "80%", margin: "auto", fontSize: "1.4rem", marginTop: "2rem" }}>
          <h3 style={subheadingStyle}>🚀 How to Improve?</h3>
          <ul style={listStyle}>
            <li><strong>Balance the Dataset:</strong> Use SMOTE (Synthetic Minority Over-sampling Technique) to improve recall for loan defaults.</li>
            <li><strong>Class Weights:</strong> Assign higher weight to Class 1 in model training to reduce bias.</li>
            <li><strong>Hyperparameter Tuning:</strong> Adjust max_depth, min_samples_split, n_estimators to optimize performance.</li>
            <li><strong>Explore Other Models:</strong> Consider XGBoost or Gradient Boosting, which handle class imbalance more effectively.</li>
          </ul>
        </div>
      </div>

      <div style={{ textAlign: "center", padding: "3rem", maxWidth: "90%", margin: "auto" }}>
        <h2 style={headingStyle}>
          🚀 Future Work
        </h2>

        {/* Grid Layout for Future Work */}
        <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr 1fr", 
            alignItems: "start", 
            gap: "3rem",
            marginBottom: "2rem"
        }}>

          {/* Exploring Deep Learning Approaches */}
          <div>
            <h3 style={subheadingStyle}>
              🧠 Exploring Deep Learning Approaches
            </h3>
            <p style={paragraphStyle}>
              We plan to enhance model performance by transitioning to deep learning methods such as 
              <strong> Artificial Neural Networks (ANNs)</strong> or <strong>Long Short-Term Memory (LSTMs)</strong>. 
              These models can improve feature learning and capture complex patterns in loan repayment behavior.
            </p>
            
          </div>

          {/* Improving Model Generalization */}
          <div>
            <h3 style={subheadingStyle}>
              🔄 Improving Model Generalization
            </h3>
            <p style={paragraphStyle}>
              Addressing <strong>class imbalance</strong> remains a priority. Future improvements will incorporate
              <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> and 
              <strong> cost-sensitive learning</strong> to ensure the model effectively captures minority class patterns.
            </p>
            
          </div>

          {/* Expanding the Dataset */}
          <div>
            <h3 style={subheadingStyle}>
              🌐 Expanding the Dataset
            </h3>
            <p style={paragraphStyle}>
              Increasing dataset size and diversity will help reduce bias. Future efforts will focus on 
              <strong> incorporating alternative financial indicators</strong> such as 
              <strong> transaction history and additional credit scoring factors</strong>.
            </p>
            
          </div>

        </div>
      </div>

      {/* GitHub Link */}
      <a href="https://github.com/mhuy26/decision-tree-and-random-forrest-Loan-Repayment-Prediction" target="_blank" rel="noopener noreferrer"
        style={{ color: "#61dafb", textDecoration: "none", fontWeight: "bold", display: "block", marginTop: "1.5rem", fontSize: "1.3rem" }}>
        View on GitHub
      </a>
    </div>
  );
}