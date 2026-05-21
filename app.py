import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document

# PAGE SETTINGS
st.set_page_config(
    page_title="Document Search Engine",
    layout="wide"
)

# CUSTOM STYLE
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1 {
    text-align: center;
    color: white;
}

.stButton button {
    width: 100%;
    background-color: orange;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.title("DOCUMENT SEARCH ENGINE")

st.write(
    "Upload TXT, PDF or DOCX files and search content dynamically using TF-IDF based NLP retrieval."
)

# COLUMNS
left_col, right_col = st.columns(2)

# FUNCTIONS
def extract_pdf_text(file):

    text = ""

    pdf = PdfReader(file)

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


def extract_docx_text(file):

    doc = Document(file)

    text = "\n".join([para.text for para in doc.paragraphs])

    return text


def extract_txt_text(file):

    return file.read().decode("utf-8")


# LEFT PANEL
with left_col:

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    query = st.text_input(
        "Enter Search Query",
        placeholder="Example: internship"
    )

    search = st.button("Submit")

# RIGHT PANEL
with right_col:

    st.subheader("Search Results")

# SEARCH LOGIC
if search:

    if uploaded_files and query:

        documents = []
        filenames = []

        # READ DOCUMENTS
        for file in uploaded_files:

            if file.type == "application/pdf":

                text = extract_pdf_text(file)

            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

                text = extract_docx_text(file)

            else:

                text = extract_txt_text(file)

            documents.append(text)

            filenames.append(file.name)

        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")

        all_text = documents + [query]

        tfidf_matrix = vectorizer.fit_transform(all_text)

        similarity = cosine_similarity(
            tfidf_matrix[-1],
            tfidf_matrix[:-1]
        )

        scores = similarity.flatten()

        ranked_results = sorted(
            zip(filenames, scores, documents),
            key=lambda x: x[1],
            reverse=True
        )

        found = False

        # DISPLAY RESULTS
        for filename, score, content in ranked_results:

            if score > 0:

                found = True

                st.markdown("---")

                st.markdown(f"### DOCUMENT:")
                st.write(filename)

                st.markdown(
                    f"### RELEVANCE SCORE: {round(score * 100, 2)}%"
                )

                st.markdown("### MATCHED CONTENT PREVIEW:")

                query_words = query.lower().split()

                matched_text = ""

                sentences = content.split(".")

                for sentence in sentences:

                    for word in query_words:

                        if word in sentence.lower():

                            matched_text += sentence.strip() + ". "

                if matched_text == "":

                    matched_text = content[:500]

                st.write(matched_text[:1000])

        if not found:

            st.warning("No matching content found.")

    else:

        st.warning("Please upload files and enter search query.")