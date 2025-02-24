import { FaPython, FaJava, FaGithub } from "react-icons/fa";
import { SiCplusplus, SiTensorflow, SiKeras, SiScikitlearn, SiJupyter } from "react-icons/si";
import { SiR } from "react-icons/si";

export default function Home() {
  const technologies = [
    { name: "Python", icon: <FaPython size={50} color="white" /> },
    { name: "C++", icon: <SiCplusplus size={50} color="white" /> },
    { name: "Java", icon: <FaJava size={50} color="white" /> },
    { name: "R", icon: <SiR size={50} color="white" /> },
    { name: "TensorFlow", icon: <SiTensorflow size={50} color="white" /> },
    { name: "Keras", icon: <SiKeras size={50} color="white" /> },
    { name: "Scikit-learn", icon: <SiScikitlearn size={50} color="white" /> },
    { name: "GitHub", icon: <FaGithub size={50} color="white" /> },
    { name: "Jupyter Notebook", icon: <SiJupyter size={50} color="white" /> },
    { name: "RStudio", icon: <SiR size={50} color="white" /> },  // Using R logo as a placeholder

  ];

  const interests = [
    { text: "Photography", emoji: "📷" },
    { text: "Cats", emoji: "🐱" },
    { text: "Playing Badminton", emoji: "🏸" },
  ];

  return (
    <div style={{ textAlign: "center", padding: "2rem" }}>
      <h1>Hello, I'm Huy /hwee/ 👋</h1>
      <p>
        I'm a Data Science student at UT Dallas, studying Data Science.
      </p>
      <p>
        I'm passionate about AI, Machine Learning, and Data Analytics.
      </p>
      <p>
        Check out my works ☝️
      </p>
      <br />

      
      <h3>Besides coding, I love:</h3>
      <ul style={{ 
        listStyleType: "none", 
        padding: 0, 
        textAlign: "center", 
        fontSize: "18px", 
        lineHeight: "1.8" 
      }}>
        {interests.map((interest, index) => (
          <li key={index} style={{ marginBottom: "5px" }}>
            {interest.text} {interest.emoji}
          </li>
        ))}
      </ul>
      <br />


      <h2>Technologies I Work With</h2>
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
        gap: "20px",
        justifyContent: "center",
        alignItems: "center",
        maxWidth: "800px",
        margin: "auto",
        paddingTop: "1rem"
      }}>
        {technologies.map((tech) => (
          <div key={tech.name} style={{
            textAlign: "center",
            padding: "10px",
            borderRadius: "10px",
            backgroundColor: "#1e1e1e",
            boxShadow: "0px 0px 10px rgba(255, 255, 255, 0.2)",
            transition: "0.3s",
            cursor: "pointer"
          }}
          onMouseOver={(e) => e.currentTarget.style.transform = "scale(1.1)"}
          onMouseOut={(e) => e.currentTarget.style.transform = "scale(1.0)"}>
            {tech.icon}
            <p style={{ fontSize: "14px", marginTop: "5px", color: "white" }}>{tech.name}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
