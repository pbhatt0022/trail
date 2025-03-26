import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_processor import extract_text_from_pdf, clean_text, extract_resume_sections
from utils.nlp_processor import analyze_job_description, analyze_resume, calculate_match_score
from utils.ui_components import display_match_score_gauge, display_keyword_match_bar, display_match_details_expander, display_recommendations
from utils.openai_helpers import initialize_openai, generate_interview_questions

# Configure page settings
st.set_page_config(page_title="AI Interview Prep", page_icon="🧠", layout="wide")

# Page title
st.title("AI Interview Preparation Platform")
st.markdown("---")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check for OpenAI API key
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in your .env file!")
    st.info("Please add your OpenAI API key to the .env file")
    st.stop()

# Initialize OpenAI client
openai_client = initialize_openai()
if not openai_client:
    st.info("Please check your .env file and ensure the OPENAI_API_KEY is set correctly.")
    st.stop()

# ---- SIDEBAR SETUP ----
st.sidebar.title("Interview Setup")
job_title = st.sidebar.text_input(
    "Job Title",
    placeholder="e.g. Senior Data Scientist",
    value=st.session_state.get("job_title", "")
)

# Update job title in session state whenever it changes
if job_title != st.session_state.get("job_title", ""):
    st.session_state["job_title"] = job_title

# ---- PAGE NAVIGATION ----
# Define available pages
pages = ["Upload Resume", "Resume-Job Match", "Generate Questions", "Interview Session"]

# Get current page from query params or default to first page
query_params = st.query_params
current_page = query_params.get("page", pages[0])
if current_page not in pages:
    current_page = pages[0]

# Update page via sidebar
page = st.sidebar.radio("Navigation", pages, index=pages.index(current_page))

# If page changes, update query params
if page != current_page:
    st.query_params["page"] = page
    current_page = page

# ---- SESSION STATE INITIALIZATION ----
# Initialize session state variables if not already present
if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = None
if "resume_sections" not in st.session_state:
    st.session_state["resume_sections"] = {}
if "job_description" not in st.session_state:
    st.session_state["job_description"] = ""
if "match_result" not in st.session_state:
    st.session_state["match_result"] = None
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None
# Flag to track if resume is uploaded and processed
if "resume_uploaded" not in st.session_state:
    st.session_state["resume_uploaded"] = False

# ---- PAGE 1: Upload Resume ----
if page == "Upload Resume":
    st.header("Resume Upload")
    st.write("Upload your resume to begin the interview preparation process.")

    # File uploader for resume
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    
    # Job description input
    job_description = st.text_area(
        "Job Description",
        value=st.session_state.get("job_description", ""),
        placeholder="Paste the job description here...",
        height=200
    )
    st.session_state["job_description"] = job_description

    # Process resume if uploaded
    if uploaded_file is not None:
        # Check if this is a new file or already processed
        if (st.session_state.get("uploaded_file_name") != uploaded_file.name) or (st.session_state.get("resume_text") is None):
            with st.spinner("Processing resume..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                if resume_text:
                    # Process the resume text
                    cleaned_text = clean_text(resume_text)
                    resume_sections = extract_resume_sections(cleaned_text)
                    
                    # Save to session state
                    st.session_state["resume_text"] = cleaned_text
                    st.session_state["resume_sections"] = resume_sections
                    st.session_state["uploaded_file_name"] = uploaded_file.name
                    st.session_state["resume_uploaded"] = True
                    
                    st.success("✅ Resume uploaded and processed successfully!")
                else:
                    st.error("Failed to extract text from the PDF. Please try another file.")
        else:
            # Already processed this file before
            st.session_state["resume_uploaded"] = True
            st.success("✅ Resume already processed!")

        # Display resume preview if text exists
        if st.session_state.get("resume_text"):
            with st.expander("Preview Extracted Resume"):
                preview_text = st.session_state["resume_text"]
                st.text_area(
                    "Resume Content",
                    value=preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text,
                    height=200,
                    disabled=True
                )

    # Navigation button
    if st.session_state.get("resume_text") and job_description:
        if st.button("Next Step: Analyze Resume-Job Match", type="primary"):
            st.query_params["page"] = "Resume-Job Match"
            st.rerun()
    else:
        st.info("Please upload a resume and enter a job description to continue.")

# ---- PAGE 2: Resume-Job Match ----
elif page == "Resume-Job Match":
    st.header("Resume and Job Description Match Analysis")

    # Check if we have a resume
    if not st.session_state.get("resume_text") or not st.session_state.get("resume_uploaded", False):
        st.warning("⚠️ Please upload your resume first!")
        if st.button("Go to Resume Upload"):
            st.query_params["page"] = "Upload Resume"
            st.rerun()
    else:
        # Display job description for review/edit
        job_description = st.text_area(
            "Job Description",
            value=st.session_state.get("job_description", ""),
            placeholder="Paste the job description here...",
            height=200
        )
        st.session_state["job_description"] = job_description

        # Job title check
        if not st.session_state.get("job_title"):
            st.warning("⚠️ Please enter a job title in the sidebar!")
        
        # Analyze button
        analyze_button = st.button("Analyze Match", disabled=not (st.session_state.get("job_title") and job_description))
        
        if analyze_button:
            with st.spinner("Analyzing your resume against the job description..."):
                # Perform analysis
                job_analysis = analyze_job_description(job_description)
                resume_analysis = analyze_resume(
                    st.session_state["resume_text"],
                    st.session_state.get("resume_sections", {})
                )
                match_result = calculate_match_score(resume_analysis, job_analysis)
                
                # Store results
                st.session_state["job_analysis"] = job_analysis
                st.session_state["resume_analysis"] = resume_analysis
                st.session_state["match_result"] = match_result
            
            st.success("✅ Analysis complete!")

        # Display results if available
        if st.session_state.get("match_result"):
            match_result = st.session_state["match_result"]
            
            st.subheader("Resume-Job Match Results")
            
            # Layout results in columns
            col1, col2 = st.columns(2)
            with col1:
                display_match_score_gauge(match_result.get("overall_score", 0), "Overall Match Score")
            
            with col2:
                display_keyword_match_bar(
                    match_result.get("matching_keywords", []),
                    match_result.get("missing_keywords", [])
                )
            
            # Display detailed match information
            display_match_details_expander(match_result)
            
            # Display recommendations
            display_recommendations(match_result, st.session_state.get("job_title", ""))
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update Resume/Job"):
                    st.query_params["page"] = "Upload Resume"
                    st.rerun()
            
            with col2:
                if st.button("Next: Generate Interview Questions", type="primary"):
                    st.query_params["page"] = "Generate Questions"
                    st.rerun()

# ---- PAGE 3: Generate Questions ----
elif page == "Generate Questions":
    st.header("Generate Interview Questions")
    
    # Verify requirements are met
    if not st.session_state.get("resume_text") or not st.session_state.get("resume_uploaded", False):
        st.warning("⚠️ Please upload your resume first!")
        if st.button("Go to Resume Upload"):
            st.query_params["page"] = "Upload Resume"
            st.rerun()
    elif not st.session_state.get("match_result"):
        st.warning("⚠️ Please analyze your resume-job match first!")
        if st.button("Go to Resume-Job Match"):
            st.query_params["page"] = "Resume-Job Match"
            st.rerun()
    else:
        # Display match summary
        col1, col2 = st.columns(2)
        with col1:
            match_score = st.session_state["match_result"].get("overall_score", 0)
            st.info(f"Overall Match Score: {match_score*100:.1f}%")
        
        with col2:
            matching_count = len(st.session_state["match_result"].get("matching_keywords", []))
            missing_count = len(st.session_state["match_result"].get("missing_keywords", []))
            st.info(f"Keywords: {matching_count} matching, {missing_count} missing")
        
        st.divider()
        
        # Generate questions button
        if st.button("Generate Interview Questions", type="primary"):
            with st.spinner("Generating personalized interview questions..."):
                questions = generate_interview_questions(
                    openai_client,
                    st.session_state.get("job_title", ""),
                    st.session_state.get("job_description", ""),
                    st.session_state.get("resume_text", ""),
                    st.session_state.get("match_result", {})
                )
                
                if questions:
                    st.session_state["questions"] = questions
                    st.success("✅ Questions generated successfully!")
                else:
                    st.error("Failed to generate questions. Please try again.")
        
        # Display questions if available
        if st.session_state.get("questions"):
            st.subheader("Your Interview Questions:")
            for i, question in enumerate(st.session_state["questions"], start=1):
                question_text = question.lstrip('- *').strip()
                st.write(f"{i}. {question_text}")
            
            # Navigation
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back to Resume-Job Match"):
                    st.query_params["page"] = "Resume-Job Match"
                    st.rerun()
            
            with col2:
                if st.button("Next: Start Interview Session", type="primary"):
                    st.query_params["page"] = "Interview Session"
                    st.rerun()

# ---- PAGE 4: Interview Session ----
elif page == "Interview Session":
    st.header("Interview Session")
    
    if not st.session_state.get("questions"):
        st.warning("⚠️ Please generate interview questions first!")
        if st.button("Go to Generate Questions"):
            st.query_params["page"] = "Generate Questions"
            st.rerun()
    else:
        # Interview preparation section
        st.subheader("Prepare for Your Interview")
        st.write("Review these questions and practice your responses. Consider recording yourself to improve your delivery.")
        
        # Display job title and match score
        if st.session_state.get("job_title") and st.session_state.get("match_result"):
            st.info(f"Position: {st.session_state.get('job_title')}")
            match_score = st.session_state.get("match_result", {}).get("overall_score", 0)
            st.progress(match_score, text=f"Resume Match: {match_score*100:.1f}%")
        
        st.divider()
        
        # Display questions as expandable sections
        st.subheader("Interview Questions:")
        for i, question in enumerate(st.session_state["questions"], start=1):
            question_text = question.lstrip('- *').strip()
            
            with st.expander(f"Question {i}: {question_text}"):
                user_answer = st.text_area(
                    "Practice your answer here:",
                    key=f"answer_{i}",
                    height=150,
                    placeholder="Type your practice answer here..."
                )
                
                # Tips section
                if user_answer:
                    st.info("💡 Tips: Be concise, use the STAR method for behavioral questions, and provide specific examples.")
        
        # Practice mode section
        st.divider()
        st.subheader("Practice Mode")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Start a timed practice session to prepare for the real interview.")
            if st.button("Start Practice Interview"):
                st.session_state["practice_mode"] = True
        
        # If practice mode is active
        if st.session_state.get("practice_mode", False):
            st.info("🎥 Practice mode activated! Imagine you're in a real interview.")
            
            # Simple timer using JavaScript
            st.markdown("""
            <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                <p>Interview practice in progress...</p>
                <p id="timer">00:00</p>
            </div>
            
            <script>
                // Simple timer script
                var seconds = 0;
                var minutes = 0;
                var timerElement = document.getElementById('timer');
                
                setInterval(function() {
                    seconds++;
                    if (seconds >= 60) {
                        seconds = 0;
                        minutes++;
                    }
                    
                    var formattedMinutes = (minutes < 10) ? "0" + minutes : minutes;
                    var formattedSeconds = (seconds < 10) ? "0" + seconds : seconds;
                    
                    timerElement.textContent = formattedMinutes + ":" + formattedSeconds;
                }, 1000);
            </script>
            """,
            unsafe_allow_html=True)
            
            # End practice button
            if st.button("End Practice"):
                st.session_state["practice_mode"] = False
                st.rerun()
        
        # Navigation button
        st.divider()
        if st.button("Back to Questions"):
            st.query_params["page"] = "Generate Questions"
            st.rerun()