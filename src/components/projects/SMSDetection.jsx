export default function SMSDetection() {
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
    <div style={{ textAlign: "center", padding: "2rem", color: "#E0E0E0" }}>
      <h2 style={headingStyle}>📩 SMS Spam Detection - NLP</h2>

      {/* Project Summary */}
      <div style={{ textAlign: "left", maxWidth: "75%", margin: "auto", paddingTop: "1rem" }}>
        <ul style={listStyle}>
          <li>Developed an <strong>SMS spam detection system</strong> using <strong>Scikit-learn, Pandas, and Numpy</strong>.</li>
          <li>Trained on <strong>5,000+ messages</strong> from the <strong>UCI SMS Spam Collection dataset</strong>.</li>
          <li>Applied <strong>text preprocessing</strong> (punctuation removal, stop words filtering, tokenization, stemming).</li>
          <li>Used <strong>Bag of Words (BoW) & TF-IDF</strong> for feature extraction.</li>
          <li>Achieved <strong>96% accuracy</strong> with a <strong>Naïve Bayes classifier</strong>.</li>
        </ul>
      </div>

      {/* EDA Section - Histogram of Message Lengths */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "80%", margin: "auto" }}>
        <p style={paragraphStyle}>
          We analyze the distribution of <strong>message lengths</strong> for spam and ham messages. The histogram below shows that <strong>spam messages tend to be longer</strong> compared to non-spam messages.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "2rem" }}>
          {/* Left: Message Length Histogram */}
          <img 
            src="/portfolio/img_sms/dist_spamham.png"  
            alt="Message Length Histogram"
            style={{ width: "100%", borderRadius: "8px" }}
          />

          {/* Right: Insights */}
          <div style={{ textAlign: "left" }}>
            <h3 style={subheadingStyle}>📊 Message Length Insights</h3>
            <ul style={listStyle}>
              <li><strong>Spam messages are generally longer</strong> than ham messages.</li>
              <li>Most ham messages are <strong>short and concise</strong>, often under <strong>100 characters</strong>.</li>
              <li>Spam messages often include <strong>marketing phrases, links, and call-to-actions</strong>, making them longer.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Large Section Title: Text Preprocessing */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "80%", margin: "auto" }}>
        <h2 style={headingStyle}>
          🛠️ Text Preprocessing
        </h2>

        {/* Steps Overview */}
        <p style={paragraphStyle}>Steps in Text Preprocessing:</p>
        <ol style={listStyle}>
          <li>Remove punctuation and stopwords</li>
          <li>Tokenize the message into a list of meaningful words</li>
          <li>Convert text into a numerical representation using Bag-Of-Words</li>
          <li>Apply TF-IDF weighting for feature importance</li>
        </ol>
      </div>

      {/* Stopwords Removal Section */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem", maxWidth: "80%", margin: "auto" }}>
        {/* Code Block */}
        <pre style={{
          backgroundColor: "#181818",
          color: "#E0E0E0",
          padding: "1.5rem",
          borderRadius: "8px",
          textAlign: "left",
          overflowX: "auto",
          fontSize: "1rem"
        }}>
          {`# Import Stopwords from NLTK
from nltk.corpus import stopwords
stopwords.words('english')[:10]

# Output: ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']`}
        </pre>

        {/* Explanation */}
        <div style={{ textAlign: "left" }}>
          <h3 style={subheadingStyle}>🛑 Removing Stopwords</h3>
          <ul style={listStyle}>
            <li>Stopwords are common words that do not contribute much meaning.</li>
            <li>Examples: "i", "me", "my", "we", "our", "you".</li>
            <li>Removing them helps the model focus on important words.</li>
          </ul>
        </div>
      </div>

      {/* Tokenization Section */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem", maxWidth: "80%", margin: "auto", marginTop: "2rem" }}>
        {/* Code Block */}
        <pre style={{
          backgroundColor: "#181818",
          color: "#E0E0E0",
          padding: "1.5rem",
          borderRadius: "8px",
          textAlign: "left",
          overflowX: "auto",
          fontSize: "1rem"
        }}>
          {`# Tokenizing SMS Messages
messages['message'].head(5).apply(text_process)

# Output Example:
# 0    [Go, jurong, point, crazy, Available, bugis]
# 1    [Ok, lar, Joking, wif, u, onli]
# 2    [Free, entry, 2, wkly, comp, win, FA, Cup]`}
        </pre>

        {/* Explanation */}
        <div style={{ textAlign: "left" }}>
          <h3 style={subheadingStyle}>🔡 Tokenization</h3>
          <ul style={listStyle}>
            <li>Converts text into individual words.</li>
            <li>Removes punctuation and stopwords.</li>
            <li>Prepares text for numerical transformation.</li>
          </ul>
        </div>
      </div>

      {/* Bag of Words (BoW) Section */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
        <pre style={{
          backgroundColor: "#181818",
          color: "#E0E0E0",
          padding: "1.5rem",
          borderRadius: "8px",
          textAlign: "left",
          overflowX: "auto",
          fontSize: "1rem"
        }}>
          {`# Bag-of-Words (BoW) Vectorization
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message']) 

# Output:
Shape of Sparse Matrix: (5572, 11444)
Amount of Non-Zero occurences: 50795`

}
        </pre>

        {/* Explanation */}
        <div style={{ textAlign: "left" }}>
          <h3 style={subheadingStyle}>🔢 Vectorization (Bag of Words)</h3>
          <ul style={listStyle}>
            <li>Converts tokenized words into a word count matrix.</li>
            <li>Uses CountVectorizer to create numerical feature vectors.</li>
            <li>The dataset contains 11,444 unique words.</li>
          </ul>
        </div>
      </div>

      {/* Why Vectorization? */}
      <p style={{ ...paragraphStyle, fontStyle: "italic", textAlign: "center", marginTop: "1.5rem", color: "#E0E0E0" }}>
        Why? 🤔  
        Machines can't understand words, but they can process numbers!  
        Vectorization transforms text into numerical form so it can be used for machine learning.
      </p>

      {/* TF-IDF Weighting Section */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
        <pre style={{
          backgroundColor: "#181818",
          color: "#E0E0E0",
          padding: "1.5rem",
          borderRadius: "8px",
          textAlign: "left",
          overflowX: "auto",
          fontSize: "1rem"
        }}>
          {`# TF-IDF Transformation
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Transform the entire bag-of-word corpus into TF-IDF corpus
messages_tfidf = tfidf_transformer.transform(messages_bow)
`}
        </pre>

        {/* Explanation */}
        <div style={{ textAlign: "left" }}>
          <h3 style={subheadingStyle}>📊 TF-IDF Transformation</h3>
          <ul style={listStyle}>
            <li>Term Frequency-Inverse Document Frequency (TF-IDF) assigns importance to words.</li>
            <li>Common words are weighted lower, while rare words are weighted higher.</li>
            <li>Each number represents the TF-IDF weight of a word in a message.</li>
            <li>Higher values indicate more important words for classification.</li>
            <li>This helps improve the model's understanding of relevant words.</li>
          </ul>
        </div>
      </div>

      {/* Why TF-IDF? */}
      <p style={{ ...paragraphStyle, fontStyle: "italic" }}>
        Why? 🤔  
        Not all words carry the same importance.  
        TF-IDF ensures that frequently occurring words get less weight, while rare but important words stand out.
      </p>




      {/* Training a Model Section */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
        <h2 style={headingStyle}>
          🤖 Training the Spam Classifier
        </h2>

        <p style={paragraphStyle}>
          Now that we have preprocessed and vectorized our SMS messages, we can train a **spam classifier** to distinguish between spam and ham messages.
          For this, we will use the **Naïve Bayes** algorithm, a popular choice for text classification.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem", marginTop: "2rem" }}>
          {/* Code Block */}
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "8px",
            textAlign: "left",
            overflowX: "auto",
            fontSize: "1rem"
          }}>
            {`# Import Naïve Bayes classifier from sklearn
from sklearn.naive_bayes import MultinomialNB

# Train the model using the TF-IDF matrix and message labels
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])`}
          </pre>

          {/* Explanation */}
          <div style={{ textAlign: "left" }}>
            <h3 style={subheadingStyle}>📌 Why Naïve Bayes?</h3>
            <ul style={listStyle}>
              <li>Computationally **fast and efficient**, even for large datasets.</li>
              <li>Works well with **high-dimensional text data** (thousands of unique words).</li>
              <li>Performs well even with **small amounts of training data**.</li>
              <li>Commonly used for **email spam filtering and text classification**.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Train-Test Split Section */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
        <h2 style={headingStyle}>🔀 Train-Test Split</h2>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem" }}>
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "8px",
            textAlign: "left",
            overflowX: "auto",
            fontSize: "1rem"
          }}>
            {`# Splitting Data into Train and Test Sets
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

# Output:
# 4457 1115 5572`}
          </pre>

          <div style={{ textAlign: "left" }}>
            <h3 style={subheadingStyle}>📊 Why Split Data?</h3>
            <ul style={listStyle}>
              <li>We divide data into training (80%) and testing (20%) sets.</li>
              <li>Training set teaches the model, while the test set evaluates its performance.</li>
              <li>This prevents overfitting and ensures the model generalizes well.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Creating a Data Pipeline */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
        <h2 style={headingStyle}>🔗 Creating a Data Pipeline</h2>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem" }}>
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "8px",
            textAlign: "left",
            overflowX: "auto",
            fontSize: "1rem"
          }}>
            {`# Building an NLP Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)), # Convert text to word count matrix
    ('tfidf', TfidfTransformer()), # Apply TF-IDF weighting
    ('classifier', MultinomialNB()), # Train a Naive Bayes classifier
])

# Train the pipeline
pipeline.fit(msg_train, label_train)`}
          </pre>

          <div style={{ textAlign: "left" }}>
            <h3 style={subheadingStyle}>🚀 Why Use a Pipeline?</h3>
            <ul style={listStyle}>
              <li>Automates text processing, vectorization, and model training.</li>
              <li>Makes predictions easier by processing raw text directly.</li>
              <li>Improves efficiency and reduces manual steps.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Model Evaluation Section */}
      <div style={{ textAlign: "left", padding: "2rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
        <h2 style={headingStyle}>📈 Model Evaluation</h2>

        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", alignItems: "center", gap: "4rem" }}>
          <pre style={{
            backgroundColor: "#181818",
            color: "#E0E0E0",
            padding: "1.5rem",
            borderRadius: "8px",
            textAlign: "left",
            overflowX: "auto",
            fontSize: "1rem"
          }}>
            {`# Evaluating Model Performance
from sklearn.metrics import classification_report

predictions = pipeline.predict(msg_test)

print(classification_report(predictions, label_test))

# Output:
# precision    recall  f1-score   support
# ham       1.00    0.96      0.98       1001
# spam      0.75    1.00      0.85       114
# avg       0.97    0.97      0.97      1115`}
          </pre>

          <div style={{ textAlign: "left" }}>
            <h3 style={subheadingStyle}>📊 Understanding the Results</h3>
            <ul style={listStyle}>
              <li><b>Precision:</b> Accuracy of spam detection.</li>
              <li><b>Recall:</b> How many actual spam messages were detected.</li>
              <li><b>F1-score:</b> Balance of precision and recall.</li>
              <li>Our model achieves an overall F1-score of 97%, meaning it is highly accurate.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Future Work & Model Improvements */}
<div style={{ textAlign: "center", padding: "3rem", maxWidth: "80%", margin: "auto", marginTop: "3rem" }}>
  <h2 style={headingStyle}>
    🚀 Future Work & Model Improvements
  </h2>

  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4rem", textAlign: "center" }}>
    
    {/* Improving Model Performance */}
    <div>
      <h3 style={subheadingStyle}>📈 Improving Model Performance</h3>
      <p style={paragraphStyle}>
        Experiment with advanced classification models such as Random Forest, XGBoost, or deep learning models like LSTMs for better accuracy.
      </p>
      <p style={{ ...paragraphStyle, fontStyle: "italic" }}>
        Fine-tune hyperparameters using Grid Search or Bayesian Optimization.
      </p>
    </div>

    {/* Expanding the Dataset */}
    <div>
      <h3 style={subheadingStyle}>📊 Expanding the Dataset</h3>
      <p style={paragraphStyle}>
        Collect and incorporate new spam datasets to enhance model generalization and reduce bias.
      </p>
      <p style={{ ...paragraphStyle, fontStyle: "italic" }}>
        Introduce multilingual spam messages to make the model more robust.
      </p>
    </div>

    {/* Real-Time Detection */}
    <div>
      <h3 style={subheadingStyle}>⏳ Real-Time Detection</h3>
      <p style={paragraphStyle}>
        Deploy the model as a live service that scans and classifies SMS messages in real-time.
      </p>
      <p style={{ ...paragraphStyle, fontStyle: "italic" }}>
        Implement it as a browser extension or mobile application.
      </p>
    </div>

    {/* Model Deployment */}
    <div>
      <h3 style={subheadingStyle}>🌐 Model Deployment</h3>
      <p style={paragraphStyle}>
        Deploy the trained model as a web API using Flask or FastAPI to serve real-time spam detection.
      </p>
      <p style={{ ...paragraphStyle, fontStyle: "italic" }}>
        Consider cloud-based deployment options such as AWS Lambda, Google Cloud, or Heroku.
      </p>
    </div>
  
  </div>
</div>



      

      {/* GitHub Link */}
      <a href="https://github.com/mhuy26/SpamOrHamDectect" target="_blank" rel="noopener noreferrer"
        style={{ color: "#61dafb", textDecoration: "none", fontWeight: "bold", display: "block", marginTop: "1rem", fontSize: "1.2rem" }}>
        View on GitHub
      </a>
    </div>
  );
}
