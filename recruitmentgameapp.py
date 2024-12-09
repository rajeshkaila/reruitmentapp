import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the questions
questions = [
    {"id": 1, "question": "What comes next in the sequence: 2, 4, 6, ?", 
     "options": ["7", "8", "9"], "answer": "8"},
    {"id": 2, "question": "If A > B and B > C, which of the following is true?", 
     "options": ["A > C", "C > A", "A = C"], "answer": "A > C"},
    {"id": 3, "question": "What is 15% of 200?", 
     "options": ["30", "25", "35"], "answer": "30"},
    {"id": 4, "question": "Which word does not belong: Dog, Cat, Elephant, Car?", 
     "options": ["Dog", "Elephant", "Car", "Cat"], "answer": "Car"},
    {"id": 5, "question": "Find the odd one out: 24, 36, 48, 60, 72, 81", 
     "options": ["48", "72", "81", "60"], "answer": "81"},
    {"id": 6, "question": "If all Bloops are Razzies, and all Razzies are Lazzies, are all Bloops definitely Lazzies?", 
     "options": ["Yes", "No", "Cannot Determine"], "answer": "Yes"},
    {"id": 7, "question": "A clock shows the time as 3:15. What is the angle between the hour and minute hands?", 
     "options": ["7.5°", "15°", "22.5°"], "answer": "7.5°"},
    {"id": 8, "question": "Which number replaces the question mark? 1, 4, 9, 16, ?, 36", 
     "options": ["20", "25", "30"], "answer": "25"},
    {"id": 9, "question": "If you have 3 apples and take away 2, how many do you have?", 
     "options": ["1", "2", "3"], "answer": "2"},
    {"id": 10, "question": "A train leaves the station at 3:00 PM traveling at 60 mph. Another train leaves the same station at 4:00 PM traveling at 80 mph. When will the second train catch up to the first?",
     "options": ["5:00 PM", "6:00 PM", "7:00 PM"], "answer": "6:00 PM"}
]

# Initialize session state variables if not already present
if "responses" not in st.session_state:
    st.session_state.responses = []

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

# Streamlit setup
st.title("Logical Reasoning Game")
st.write("Test your logical reasoning skills by answering the questions below.")

# Initialize an empty DataFrame for results
response_data = pd.DataFrame(st.session_state.responses)

# Question display
if st.session_state.current_question < len(questions):
    q = questions[st.session_state.current_question]
    st.write(f"### Question {q['id']}: {q['question']}")
    options = q["options"]

    user_answer = st.radio("Choose an answer:", options, key=f"q{q['id']}")

    if st.button("Submit Answer"):
        time_taken = np.random.uniform(2, 15)  # Simulating response time
        correct = int(user_answer == q["answer"])

        # Save response
        st.session_state.responses.append({
            "question_id": q["id"],
            "time_taken": time_taken,
            "correct": correct
        })
        st.session_state.current_question += 1
        st.rerun()  # Refresh the page to load the next question
else:
    st.write("### Game Over! Here are your results:")
    response_data = pd.DataFrame(st.session_state.responses)  # Ensure we have the latest data
    if not response_data.empty:
        st.write(response_data)

    # Generate synthetic training data
    np.random.seed(42)
    train_data = pd.DataFrame({
        "question_id": np.random.randint(1, 11, 50),  # Random question IDs (1-10)
        "time_taken": np.random.uniform(2, 15, 50),  # Random time taken (2 to 15 seconds)
        "correct": np.random.choice([0, 1], 50)      # Random correctness (0 or 1)
    })

    # Generate overall skill levels based on correctness and time taken
    train_data["overall_skill_level"] = np.where(
        (train_data["correct"] == 1) & (train_data["time_taken"] < 10), "Yes", "No"
    )

    # Train/Test Split
    X_train = train_data[["question_id", "time_taken", "correct"]]
    y_train = train_data["overall_skill_level"]

    # Train Random Forest Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions on current response data
    if not response_data.empty:
        X_new = response_data[["question_id", "time_taken", "correct"]]
        response_data["overall_skill_pred"] = model.predict(X_new)

        st.write("### Predicted Overall Skill Level (Yes/No):")
        st.write(response_data)

        # Recruitment Decision Logic
        if response_data["overall_skill_pred"].iloc[-1] == "Yes":
            st.write("### Congratulations! You are recommended for recruitment!")
        else:
            st.write("### Unfortunately, you are not recommended for recruitment.")

    # Evaluate model performance on synthetic data
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    y_pred_split = model.predict(X_test_split)
    st.write(f"Model Training Accuracy: {accuracy_score(y_test_split, y_pred_split):.2f}")

    st.write("Thank you for playing!")

    # Save the responses to an Excel file
    if not response_data.empty:
        file_name = "candidate_responses.xlsx"
        response_data.to_excel(file_name, index=False)
        
        # Provide the download link
        st.download_button(
            label="Download Response Data (Excel)",
            data=open(file_name, "rb").read(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
