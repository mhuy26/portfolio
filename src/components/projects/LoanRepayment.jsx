export default function LoanRepayment() {
  return (
    <div style={{ textAlign: "center", padding: "2rem" }}>
        <h2>Keras Deep Learning Loan Repayment</h2>

        {/* Project Summary - Bullet Points */}
        <div style={{ textAlign: "left", maxWidth: "70%", margin: "auto", paddingTop: "1rem" }}>
          <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
            <li>Developed a deep learning model to predict loan repayment outcomes using LendingClub data.</li>
            <li>Performed data cleaning, handled missing values, and analyzed feature correlations for better predictions.</li>
            <li>Used Keras and TensorFlow to build a neural network optimized with dropout, batch normalization, and ReLU activation.</li>
          </ul>
        </div>




        {/* EDA Section - Image on Left, Insights on Right */}
        {/* Guidance Sentence */}
        <p style={{ fontSize: "1.1rem", marginBottom: "1rem", textAlign: "left"  }}>
        To understand the distribution of loan outcomes, we first analyze the loan_status feature. 
        The countplot below visualizes the number of fully paid vs. charged-off loans
          </p>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "2rem" }}>
          {/* Left: Countplot Image */}
          <img 
            src="/portfolio/img_loanKera/countplot.png"  // ✅ Updated Image Path
            alt="Loan Status Countplot" 
            style={{ width: "45%", borderRadius: "8px", marginRight: "2rem" }} 
          />

          {/* Right: Insights */}
          <div style={{ width: "50%", textAlign: "left" }}>
            <h3>Loan Status Distribution</h3>
            <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
              <li>The dataset is <strong>imbalanced</strong>, with significantly more fully paid loans than charged-off ones.</li>
              <li>This imbalance may affect model predictions, requiring techniques like <strong>resampling</strong> or <strong>class weighting</strong> to ensure fair learning.</li>
              <li>The presence of defaults confirms that loan repayment is not guaranteed, reinforcing the need for a <strong>risk assessment model</strong>.</li>
            </ul>
          </div>

        </div>



        {/* Loan Amount Histogram Section */}
        <div style={{ textAlign: "center", padding: "2rem" }}>
          {/* Guidance Sentence */}
          <p style={{ fontSize: "1.1rem", marginBottom: "1rem", textAlign: "left"}}>
            Next, we analyze the distribution of loan amounts to understand common borrowing patterns. 
            The histogram below visualizes the frequency of different loan amounts in the dataset.
          </p>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
            {/* Left: Loan Amount Histogram Image */}
            <img 
              src="/portfolio/img_loanKera/hist_loanAmount.png"  
              alt="Loan Amount Histogram"
              style={{ width: "45%", borderRadius: "8px", marginRight: "2rem" }}
            />

            {/* Right: Insights */}
            <div style={{ width: "50%", textAlign: "left" }}>
              <h3>Loan Amount Distribution</h3>
              <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
                <li>Most loans are clustered around certain amounts (e.g., $5,000, $10,000, $15,000), indicating standard loan packages.</li>
                <li>The distribution may show peaks at specific values, suggesting common borrowing preferences.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Correlation Heatmap Section */}
          <div style={{ textAlign: "center", padding: "2rem" }}>
          {/* Guidance Sentence */}
          <p style={{ fontSize: "1.1rem", marginBottom: "1rem",textAlign: "left"  }}>
            To explore relationships between numerical features, we generate a correlation heatmap. 
            This visualization helps identify which factors are most associated with loan repayment behavior.
          </p>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
            {/* Left: Correlation Heatmap Image */}
            <img 
              src="/portfolio/img_loanKera/heatmap_corr.png"  
              alt="Correlation Heatmap"
              style={{ width: "45%", borderRadius: "8px", marginRight: "2rem" }}
            />

            {/* Right: Insights */}
              <div style={{ width: "50%", textAlign: "left" }}>
                <h3>Strong Positive Correlations</h3>
                <ul style={{ listStyleType: "disc", paddingLeft: "20px" }}>
                  <li><strong>Loan Amount & Installment (0.95):</strong> Higher loan amounts lead to higher monthly installment payments.</li>
                  <li><strong>Total Accounts & Open Accounts (0.68):</strong> Borrowers with more total accounts tend to have more open accounts.</li>
                  <li><strong>Public Records & Bankruptcies (0.7):</strong> If a borrower has public records, it's likely they also have bankruptcies.</li>
                </ul>
              </div>
          </div>
        </div>



        {/* Data Cleaning & Preprocessing Section */}
        <div style={{ textAlign: "center", padding: "2rem" }}>
          <h2 style={{ color: "#ffffff", marginBottom: "1rem" }}>🛠️ Data Cleaning & Preprocessing</h2>

          {/* Adjusted Grid for Better Spacing */}
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            
            {/* 1️⃣ Filling Missing mort_acc Values */}
            <pre style={{
              backgroundColor: "#181818",
              color: "#d1d1d1",
              padding: "1.5rem",
              borderRadius: "8px",
              textAlign: "left",
              width: "100%",
              overflowX: "auto",
              fontSize: "0.9rem"
            }}>
                {`total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

                def fill_mort_acc(total_acc, mort_acc):
                    if np.isnan(mort_acc):
                        return total_acc_avg[total_acc]
                    else:
                        return mort_acc

                df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)`}
            </pre>

            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>🔧 Filling Missing <span style={{ color: "#ffcc00" }}>mort_acc</span> Values</h3>
              <p>Instead of dropping <span style={{ color: "#ffcc00" }}>mort_acc</span>, we filled missing values using the average based on <span style={{ color: "#ffcc00" }}>total_acc</span>.</p>
              <p>This method ensures logical imputation, as <span style={{ color: "#ffcc00" }}>mort_acc</span> and <span style={{ color: "#ffcc00" }}>total_acc</span> are correlated.</p>
            </div>




            {/* 2️⃣ Encoding Categorical Variables */}
            <pre style={{
              backgroundColor: "#181818",
              color: "#d1d1d1",
              padding: "1.5rem",
              borderRadius: "8px",
              textAlign: "left",
              width: "100%",
              overflowX: "auto",
              fontSize: "0.9rem"
            }}>
                {`subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
                df = pd.concat([df.drop('sub_grade', axis=1), subgrade_dummies], axis=1)

                dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']], drop_first=True)
                df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1)
                df = pd.concat([df, dummies], axis=1)`}
            </pre>

            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>📌 Encoding Categorical Variables</h3>
              <p>Converted <span style={{ color: "#ffcc00" }}>sub_grade</span> into dummy variables to prevent issues with categorical encoding.</p>
              <p>Used one-hot encoding for <span style={{ color: "#ffcc00" }}>verification_status</span>, <span style={{ color: "#ffcc00" }}>application_type</span>, <span style={{ color: "#ffcc00" }}>initial_list_status</span>, and <span style={{ color: "#ffcc00" }}>purpose</span>.</p>
            </div>




            {/* 3️⃣ Feature Engineering: Processing earliest_cr_line */}
            <pre style={{
              backgroundColor: "#181818",
              color: "#d1d1d1",
              padding: "1.5rem",
              borderRadius: "8px",
              textAlign: "left",
              width: "100%",
              overflowX: "auto",
              fontSize: "0.9rem"
            }}>
                {`df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
                df = df.drop('earliest_cr_line', axis=1)`}
            </pre>

            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>🔍 Feature Engineering: Extracting Year from <span style={{ color: "#ffcc00" }}>earliest_cr_line</span></h3>
              <p>Converted <span style={{ color: "#ffcc00" }}>earliest_cr_line</span> into a numerical feature (year only).</p>
              <p>This allows the model to capture the age of a credit line without using a raw timestamp.</p>
            </div>

            {/* 🚨 Drop Columns Section */}
              <img 
                src="/portfolio/img_loanKera/distribution_emp_length.png" 
                alt="Distribution of Loan Repayment by Employment Length" 
                style={{ width: "100%", borderRadius: "8px" }} 
              />
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>🚨 Dropping Unnecessary Columns</h3>
              <p>❌ Dropped <span style={{ color: "#ffcc00" }}>emp_title</span> due to too many unique values (~50% unique titles).</p>
              <p>❌ Dropped <span style={{ color: "#ffcc00" }}>emp_length</span> since all employment lengths had a ~20% likelihood of full repayment (not a useful feature).</p>
              <p>❌ Dropped <span style={{ color: "#ffcc00" }}>title</span> because it was just a subcategory of <span style={{ color: "#ffcc00" }}>purpose</span>.</p>
            </div>
          </div>




          {/* ✅ Data Cleaning Completed */}
          <div style={{ textAlign: "center", marginTop: "2rem" }}>
            <h3 style={{ color: "#00FF00" }}>✅ Data Cleaning Completed</h3>
            <p style={{ fontSize: "1rem", maxWidth: "70%", margin: "auto", color: "#d1d1d1" }}>
              We have successfully cleaned and processed the dataset, filling in missing values, dropping unnecessary features, 
              encoding categorical variables, and engineering new features for better model performance.  
              
              Now, we are ready to train our deep learning model!
            </p>
          </div>
        </div>




        /* Training & Evaluating the Deep Learning Model Section */
        <div style={{ textAlign: "center", padding: "2rem" }}>
          <h2 style={{ color: "#ffffff", marginBottom: "1rem" }}>🚀 Training & Evaluating the Deep Learning Model</h2>

          {/* Adjusted Grid for Better Spacing */}
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            
            {/* 1️⃣ Data Splitting & Scaling */}
            <pre style={{
              backgroundColor: "#181818",
              color: "#d1d1d1",
              padding: "1.5rem",
              borderRadius: "8px",
              textAlign: "left",
              width: "100%",
              overflowX: "auto",
              fontSize: "0.9rem"
            }}>
                {`from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import MinMaxScaler

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)`}
            </pre>
            
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>📊 Data Splitting & Scaling</h3>
              <p>Splitting the dataset into training (80%) and testing (20%) to ensure fair evaluation.</p>
              <p>Applied <i>MinMaxScaler</i> to normalize data, improving neural network training.</p>
            </div>
          </div>
          
          <div style={{ textAlign: "center", marginTop: "1rem", color: "#d1d1d1" }}>
            <p><i>🔍 Why? Scaling helps models converge faster & prevents dominance by large values.</i></p>
          </div>




          {/* 2️⃣ Model Architecture */}
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            <pre style={{
              backgroundColor: "#181818",
              color: "#d1d1d1",
              padding: "1.5rem",
              borderRadius: "8px",
              textAlign: "left",
              width: "100%",
              overflowX: "auto",
              fontSize: "0.9rem"
            }}>
                {`from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

                # input layer
                model.add(Dense(78,  activation='relu'))
                model.add(Dropout(0.2))

                # hidden layer
                model.add(Dense(39, activation='relu'))
                model.add(Dropout(0.2))

                # hidden layer
                model.add(Dense(19, activation='relu'))
                model.add(Dropout(0.2))

                # output layer
                model.add(Dense(units=1,activation='sigmoid'))

                # Compile model
                model.compile(loss='binary_crossentropy', optimizer='adam')
                ])`}
            </pre>
            
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>🛠️ Model Architecture</h3>
              <p>Designed a <i>Sequential</i> deep learning model with dense layers.</p>
              <p>Utilized <i>ReLU activation</i>, <i>Batch Normalization</i>, and <i>Dropout layers</i> to optimize training.</p>
            </div>
          </div>
          
          <div style={{ textAlign: "center", marginTop: "1rem", color: "#d1d1d1" }}>
            <p><i>🔍 Why? Dropout prevents overfitting; Batch Normalization stabilizes training.</i></p>
          </div>




          {/* 3️⃣ Model Compilation & Training */}
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            <pre style={{
              backgroundColor: "#181818",
              color: "#d1d1d1",
              padding: "1.5rem",
              borderRadius: "8px",
              textAlign: "left",
              width: "100%",
              overflowX: "auto",
              fontSize: "0.9rem"
            }}>
                {`model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

                history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)`}
            </pre>
            
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>⚡ Model Compilation & Training</h3>
              <p>Compiled model using <i>Binary Cross-Entropy</i> loss & <i>Adam Optimizer</i>.</p>
              <p>Trained for 25 epochs with a batch size of 32.</p>
            </div>
          </div>
          
          <div style={{ textAlign: "center", marginTop: "1rem", color: "#d1d1d1" }}>
            <p><i>🔍 Why? Binary Cross-Entropy is ideal for classification; Adam speeds up convergence while maintaining stability.</i></p>
          </div>

          {/* 4️⃣ Model Evaluation */}
          <div style={{ textAlign: "center", padding: "2rem" }}>
            <h2 style={{ color: "#ffffff", marginBottom: "1rem" }}>📊 Model Evaluation & Performance</h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            <img src="/portfolio/img_loanKera/validation_vs_training_loss.png" alt="Validation vs Training Loss" style={{ width: "100%", borderRadius: "8px" }} />
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>📉 Training vs Validation Loss</h3>
              <p>Both training loss and validation loss are dropping, but eventually won't improving that much on validation loss</p>
              <p>Maybe try adding early stop and callback and train for more epochs to see if validation loss will continue decreasing or not</p>
            </div>
          </div>

        



        {/* 5️⃣ Classification Report & Confusion Matrix */}
        <div style={{ textAlign: "center", padding: "2rem" }}>
            <h2 style={{ color: "#ffffff", marginBottom: "1rem" }}>📊 Classification Report & Confusion Matrix</h2>
          </div>


          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            <img src="/portfolio/img_loanKera/classification_report.png" alt="Classification Report" style={{ width: "100%", borderRadius: "8px" }} />
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>📋 Model Performance Insights</h3>
              <p>The dataset is highly imbalanced, with ~80% of loans being fully paid. While the model achieves 89% accuracy, this is not necessarily impressive given the imbalance.</p>
              <p>The F1-score for class <b>0</b> (charged-off loans) is only <b>0.61</b>, indicating the model struggles with predicting defaults.</p>
              <p>Future improvements include hyperparameter tuning, adding more layers, increasing neurons, or adjusting dropout rates.</p>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
            <img src="/portfolio/img_loanKera/confusion_matrix.png" alt="Confusion Matrix" style={{ width: "100%", borderRadius: "8px" }} />
            <div style={{ textAlign: "left", paddingLeft: "1rem", color: "#d1d1d1" }}>
              <h3 style={{ color: "#ffffff" }}>📊 Confusion Matrix Analysis</h3>
              <p>The confusion matrix highlights the high number of false negatives in class 0, indicating that many charged-off loans were incorrectly predicted as fully paid.</p>
              <p>Further model tuning and handling class imbalance techniques (e.g., SMOTE, cost-sensitive learning) could enhance model performance.</p>
            </div>
          </div>
        </div>




        {/* 6️⃣ Future Work & Improvements */}
        <div style={{ textAlign: "center", padding: "2rem" }}>
          <h2 style={{ color: "#ffffff", marginBottom: "1rem" }}>🚀 Future Work & Model Improvements</h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", alignItems: "center", gap: "3rem", padding: "2rem", marginLeft: "-2cm" }}>
          
          {/* 1️⃣ Hyperparameter Tuning */}
          <div style={{ textAlign: "center", padding: "1rem", color: "#d1d1d1" }}>
            <h3 style={{ color: "#ffffff" }}>🎛️ Model Tuning</h3>
            
            <p>Optimize hyperparameters such as learning rate, batch size, and number of layers to improve performance.</p>
            <p>Explore advanced techniques like <i>Bayesian Optimization</i> or <i>Grid Search</i> for better tuning.</p>
          </div>

          {/* 2️⃣ Model Deployment */}
          <div style={{ textAlign: "center", padding: "1rem", color: "#d1d1d1" }}>
            <h3 style={{ color: "#ffffff" }}>🌐 Model Deployment</h3>
            <p>Deploy the trained model as a web API using <i>Flask</i> or <i>FastAPI</i> to serve predictions in real-time.</p>
            <p>Consider cloud-based deployment options such as <i>AWS Lambda, Google Cloud, or Heroku</i>.</p>
          </div>
        </div>




        {/* GitHub Link */}
        <a href="https://github.com/mhuy26/keras_loan" target="_blank" rel="noopener noreferrer"
          style={{ color: "#61dafb", textDecoration: "none", fontWeight: "bold", display: "block", marginTop: "1rem" }}>
          View on GitHub →
        </a>
    </div>
  );
}
