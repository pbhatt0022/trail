import os
import json
from typing import List, Dict, Any
from openai import OpenAI

# Initialize the OpenAI client
def initialize_openai(api_key=None):
    """
    Initialize the OpenAI client with API key from environment variables
    
    Args:
        api_key: Optional API key to use instead of environment variable
        
    Returns:
        OpenAI client or None if initialization fails
    """
    try:
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY is not set in your .env file!")
                return None
        
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {str(e)}")
        return None

def generate_interview_questions(
    client: OpenAI,
    job_title: str, 
    job_description: str, 
    resume_text: str, 
    match_result: Dict[str, Any]
) -> List[str]:
    """
    Generate interview questions based on resume and job description using GPT-4o
    
    Args:
        client: OpenAI client
        job_title: Job title
        job_description: Job description text
        resume_text: Resume text
        match_result: Match analysis result
        
    Returns:
        List of generated interview questions
    """
    if not client:
        print("OpenAI client not initialized")
        return []
    
    # Extract missing skills to focus questions on
    missing_keywords = match_result.get("missing_keywords", [])
    missing_skills_prompt = ""
    if missing_keywords:
        missing_skills_prompt = f"""
        The candidate's resume is missing these key skills: {', '.join(missing_keywords[:5])}.
        Include questions that can assess their knowledge in these areas even if not explicitly mentioned in the resume.
        """
    
    prompt = f"""
    You are a professional interviewer conducting a job interview for the position of {job_title}.

    **Job Description:** 
    {job_description}

    **Candidate's Resume:** 
    {resume_text}

    **Resume-Job Match Analysis:**
    - Overall match score: {match_result.get('overall_score', 0)*100:.1f}%
    - Matching keywords: {', '.join(match_result.get('matching_keywords', [])[:5])}
    {missing_skills_prompt}

    **Generated Questions:**
    - Ask two technical questions related to job skills, focusing on areas where the candidate has experience.
    - Ask one technical question related to a skill that might be missing from the resume but required in the job.
    - Ask one behavioral question related to teamwork, problem-solving, or leadership.
    - Add one question to assess culture fit for the position.

    **Example Interview Questions (Few-Shot CoT):**
    **Example 1:**
    - Given that the role requires Python expertise, how would you optimize a function that runs slower than expected?
    - If you were debugging a machine learning model that was overfitting, what steps would you take?
    - Tell me about a time when you had to work with a difficult team member. How did you handle the situation?
    
    **Example 2:**
    - How would you implement a caching mechanism in a web application?
    - You notice a major security vulnerability in production. How would you address it?
    - Can you describe a situation where you had to quickly learn a new technology to complete a project?
    
    Format each question with a bullet point and ensure they are tailored to this specific job and candidate.
    """
    
    try:
        # Use the newest OpenAI model (gpt-4o) which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert interviewer who generates tailored interview questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        # Extract the generated questions
        if response.choices and response.choices[0].message.content:
            questions_text = response.choices[0].message.content
            questions = questions_text.split("\n")
            return [q for q in questions if q.strip() and (q.strip().startswith("-") or q.strip().startswith("*"))]
        else:
            return []
            
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return []