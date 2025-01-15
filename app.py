import nltk
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
from nltk.corpus import stopwords
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle
from tkinter import *
from tkinter import messagebox

# Download NLTK resources
nltk.download("stopwords")

# Load Dataset
data = pd.read_csv("dataset.csv")
print(data.head())
print(data['label'].value_counts())

# Clean text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

data["source_text"] = data["source_text"].apply(preprocess_text)
data["plagiarized_text"] = data["plagiarized_text"].apply(preprocess_text)

# Vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data["source_text"] + " " + data["plagiarized_text"])
y = data["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Model
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# Save SVM Model and Vectorizer
pickle.dump(model, open("model.pkl", 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Load Model and Vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Detection system
def detect(input_text):
    processed_text = preprocess_text(input_text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    result = model.predict(vectorized_text)
    
    # Determine plagiarized words
    plagiarized_words = []
    if result[0] == 1:  # Plagiarism detected
        plagiarized_words = processed_text.split()
    
    # Set plagiarism percentage
    if result[0] == 1:
        if hasattr(model, "decision_function"):
            score = model.decision_function(vectorized_text)[0]
            plagiarism_percentage = round((1 / (1 + abs(score))) * 100, 2)
        else:
            score = model.predict_proba(vectorized_text)[0][1]
            plagiarism_percentage = round(score * 100, 2)
        result_text = "Plagiarism Detected"
    else:
        plagiarism_percentage = 0.0
        result_text = "No Plagiarism"
    
    return result_text, plagiarism_percentage, plagiarized_words

# Highlighting function
def highlight_text(input_text, plagiarized_words):
    text_input.tag_remove("plagiarized", "1.0", END)
    text_input.tag_remove("original", "1.0", END)
    
    words = input_text.split()
    start = "1.0"
    
    for word in words:
        end = f"{start}+{len(word)}c"
        if word.lower() in plagiarized_words:
            text_input.tag_add("plagiarized", start, end)
        else:
            text_input.tag_add("original", start, end)
        start = f"{end}+1c"  # Move to the next word

# UI Function to handle input and display output
def check_plagiarism():
    input_text = text_input.get("1.0", END).strip()
    if not input_text:
        messagebox.showwarning("Input Required", "Please enter some text to check.")
        return

    result, percentage, plagiarized_words = detect(input_text)
    result_label.config(text=f"Result: {result}")
    percentage_label.config(text=f"Plagiarism Percentage: {percentage}%")
    highlight_text(input_text, plagiarized_words)

# Set up Tkinter UI
root = Tk()
root.title("Plagiarism Detection System")
root.geometry("600x400")

# Text input field with tags for color
Label(root, text="Enter Text to Check for Plagiarism:", font=("Arial", 12)).pack(pady=10)
text_input = Text(root, height=10, width=70)
text_input.pack(pady=5)
text_input.tag_configure("plagiarized", foreground="red")
text_input.tag_configure("original", foreground="green")

# Check button
check_button = Button(root, text="Check for Plagiarism", command=check_plagiarism, font=("Arial", 12), bg="blue", fg="white")
check_button.pack(pady=20)

# Result and percentage labels
result_label = Label(root, text="Result:", font=("Arial", 12))
result_label.pack()
percentage_label = Label(root, text="Plagiarism Percentage:", font=("Arial", 12))
percentage_label.pack()

# Run the UI
root.mainloop()
