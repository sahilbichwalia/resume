import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import string
import numpy as np

# Load pre-trained model and TF-IDF vectorizer (for both functionalities)
@st.cache_resource
def load_resources():
    # Job-Resume Matching Model
    matching_model = pickle.load(open('xgb.pkl', 'rb'))  # Job-Resume matching model
    tfidf_matching = pickle.load(open('tfidf1.pkl', 'rb'))
    lb_matching = pickle.load(open('encoder1.pkl', 'rb'))

    # Resume Category Prediction Model
    category_model = pickle.load(open('clf.pkl', 'rb'))  # Resume category model
    tfidf_category = pickle.load(open('tfidf.pkl', 'rb'))  # TF-IDF for category prediction
    lb_category = pickle.load(open('encoder.pkl', 'rb'))  # Label encoder for category prediction
    
    return matching_model, tfidf_matching, lb_matching, category_model, tfidf_category, lb_category


# Text cleaning function
def clean_text(txt):
    # Remove URLs
    pattern = re.compile(r'https?://\S+|www\.\S+')
    txt = pattern.sub(r'', txt)
    
    # Remove RT and cc
    txt = re.sub(r'\b(RT|cc)\b', ' ', txt)
    
    # Remove Hashtags
    txt = re.sub(r'#\S+\s', '', txt)
    
    # Remove Mentions
    txt = re.sub(r'@\S+', '', txt)
    
    # Remove Punctuation
    exclude = string.punctuation
    txt = txt.translate(str.maketrans('', '', exclude))
    
    # Remove Non-ASCII Characters
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    
    # Normalize Whitespace
    txt = re.sub('\s+', ' ', txt).strip()
    
    return txt


# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''.join(page.extract_text() for page in pdf_reader.pages)
    return text


# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text


# Extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Handle file upload and extract text
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Predict the compatibility between a resume and a job description (Job-Resume Matching)
def predict_match(resume_text, job_desc_text, model, tfidf, lb):
    # Preprocess texts
    cleaned_resume = clean_text(resume_text)
    cleaned_job_desc = clean_text(job_desc_text)

    # Vectorize texts
    vectorized_resume = tfidf.transform([cleaned_resume]).toarray()
    vectorized_job_desc = tfidf.transform([cleaned_job_desc]).toarray()

    # Combine features
    combined_features = np.concatenate((vectorized_resume, vectorized_job_desc), axis=1)

    # Make prediction
    predicted_label = model.predict(combined_features)

    # Decode the predicted label
    return lb.inverse_transform(predicted_label)[0]


# Function to predict the category of a resume (Resume Category Prediction)
def pred(input_resume, model, tfidf, lb):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = clean_text(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = lb.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# Streamlit App
def main():
    st.set_page_config(page_title="Job-Resume Matching and Category Prediction", page_icon="📄", layout="wide")
    st.title("Job Description and Resume Matching & Category Prediction")
    st.markdown("Upload a resume and provide a job description to predict whether they are a good match, or predict the job category of the resume.")

    # Load resources
    matching_model, tfidf_matching, lb_matching, category_model, tfidf_category, lb_category = load_resources()

    # Input: Job Description for Matching
    st.subheader("Enter Job Description")
    job_desc = st.text_area("Paste the job description here:", height=200)

    # Input: Resume File Upload
    uploaded_file = st.file_uploader("Upload a Resume (PDF, DOCX, or TXT format):", type=["pdf", "docx", "txt"])

    if uploaded_file:
        # Extract text from the uploaded resume file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Show extracted text (optional)
            if st.checkbox("Show extracted resume text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Job-Resume Matching Prediction
            if job_desc:
                st.subheader("Job-Resume Matching Result")
                prediction = predict_match(resume_text, job_desc, matching_model, tfidf_matching, lb_matching)
                st.success(f"The compatibility between the job description and the resume is: **{prediction}**")

            # Resume Category Prediction
            st.subheader("Resume Category Prediction")
            category = pred(resume_text, category_model, tfidf_category, lb_category)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    else:
        st.warning("Please upload a resume.")

if __name__ == "__main__":
    main()
