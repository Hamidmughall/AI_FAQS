from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS  # For handling CORS issues

# Initialize Flask App
app = Flask(__name__)

# Enable CORS (allow requests from different origins)
CORS(app)

# FAQ Data
faq_data = [
    {"question": "How do I apply for university?", "answer": "To apply for university, visit the admissions page on our website and submit your application form along with required documents."},
    {"question": "What are the eligibility requirements for admission?", "answer": "The eligibility requirements vary by program. Please check the program-specific requirements on our admissions page."},
    {"question": "When is the last date to apply for the university?", "answer": "The application deadline for the fall semester is June 30th, and for the spring semester, it is November 15th."},
    {"question": "What documents do I need for the application?", "answer": "You will need your academic transcripts, identification documents, recommendation letters, and any other program-specific documents."},
    {"question": "Can I apply if I have not completed my high school?", "answer": "No, you need to have completed your high school education or equivalent to be eligible for undergraduate programs."},
    {"question": "Do I need to provide proof of English proficiency?", "answer": "Yes, if your primary language is not English, you need to provide proof of English proficiency through tests like TOEFL or IELTS."},
    {"question": "How can I check the status of my application?", "answer": "You can check the status of your application by logging into the student portal with your credentials."},
    {"question": "What is the fee structure for international students?", "answer": "International students have different fee structures. Please refer to the international students' page for detailed information on fees."},
    {"question": "Is there any scholarship available for international students?", "answer": "Yes, we offer a range of scholarships for international students. Please visit the scholarships page for more details."},
    {"question": "What are the accommodation options for students?", "answer": "We offer on-campus dormitories as well as off-campus housing options. You can find more details on our housing page."}
]

# Preprocess Questions
questions = [faq["question"] for faq in faq_data]
answers = [faq["answer"] for faq in faq_data]

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [user_query])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    # Find Best Match
    max_index = similarities.argmax()
    if similarities[max_index] > 0.3:  # Threshold
        response = answers[max_index]
    else:
        response = "Sorry, I couldn't understand your query. Could you rephrase it?"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
