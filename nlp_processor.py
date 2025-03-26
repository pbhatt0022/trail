from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import nltk
nltk.data.path.append(r'C:\Users\priya\AppData\Roaming\nltk_data')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    # Add sentencizer to the pipeline if not present
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')
except OSError:
    # Fallback to blank model with sentencizer if model not found
    nlp = spacy.blank("en")
    nlp.add_pipe('sentencizer')

def clean_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return ""

    # Remove excessive whitespace while preserving newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # Keep more technical characters and symbols
    text = re.sub(r'[^\w\s.,;:!?()\-+#/@\[\]{}]', ' ', text)

    # Normalize line breaks but preserve paragraph structure
    text = text.replace('\r\n', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def analyze_job_description(text):
    """Analyze job description to extract key information."""
    if not text:
        print("Warning: Empty job description text")
        return {'skills': [], 'requirements': [], 'responsibilities': []}

    text = clean_text(text)
    doc = nlp(text)

    # Extract key requirements
    skills = extract_skills(text)  # Extract from full text
    requirements = []
    responsibilities = []

    for sent in doc.sents:
        sent_text = sent.text.lower()

        # Identify requirements
        if any(keyword in sent_text for keyword in ['required', 'must have', 'minimum', 'qualification']):
            requirements.append(sent.text.strip())

        # Identify responsibilities
        if any(keyword in sent_text for keyword in ['responsible for', 'duties', 'will be', 'role includes']):
            responsibilities.append(sent.text.strip())

    print(f"Found {len(skills)} skills in job description")
    return {
        'skills': skills,
        'requirements': requirements,
        'responsibilities': responsibilities
    }

def analyze_resume(text, sections):
    """Analyze resume content and extract relevant information."""
    if not text:
        print("Warning: Empty resume text")
        return {'skills': [], 'experience': [], 'education': []}

    text = clean_text(text)

    # Extract skills from both full text and specific sections
    all_skills = set()

    # Extract from full text
    full_text_skills = extract_skills(text)
    all_skills.update(full_text_skills)

    # Extract from specific sections if available
    if sections.get('skills'):
        skills_section_skills = extract_skills(sections['skills'])
        all_skills.update(skills_section_skills)

    print(f"Found {len(all_skills)} total skills in resume")

    return {
        'skills': list(all_skills),
        'experience': extract_experience(sections.get('experience', '')),
        'education': extract_education(sections.get('education', ''))
    }

def calculate_match_score(resume_analysis, job_analysis):
    """Calculate match score between resume and job description."""
    # Calculate skill match
    job_skills = set(s.lower() for s in job_analysis['skills'])
    resume_skills = set(s.lower() for s in resume_analysis['skills'])

    print(f"Job skills: {job_skills}")
    print(f"Resume skills: {resume_skills}")

    matching_skills = job_skills.intersection(resume_skills)
    missing_skills = job_skills - resume_skills

    skill_match_score = len(matching_skills) / len(job_skills) if job_skills else 0

    # Calculate overall match score (weighted average)
    overall_score = skill_match_score  # Can be expanded with more metrics

    return {
        'overall_score': overall_score,
        'skill_match_score': skill_match_score,
        'matching_keywords': list(matching_skills),
        'missing_keywords': list(missing_skills)
    }

def extract_skills(text):
    """Extract skills from text using keyword matching and NLP."""
    if not text:
        return []

    # Common technical skills and keywords
    skill_patterns = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
        'rust', 'go', 'scala', 'perl', 'r', 'matlab', 'sql', 'bash', 'shell', 'powershell',

        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask',
        'asp.net', 'spring', 'laravel', 'jquery', 'bootstrap', 'tailwind', 'webpack', 'redux',

        # Databases
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'firebase',
        'oracle', 'cassandra', 'dynamodb', 'graphql', 'nosql', 'sqlite',

        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'terraform',
        'ansible', 'puppet', 'chef', 'circleci', 'prometheus', 'grafana', 'nginx', 'apache',

        # AI/ML
        'machine learning', 'deep learning', 'neural networks', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy', 'opencv', 'nlp', 'computer vision', 'ai',

        # Other Technologies
        'rest api', 'microservices', 'git', 'agile', 'scrum', 'jira', 'confluence',
        'linux', 'unix', 'ci/cd', 'oauth', 'jwt', 'api', 'sdk', 'saas', 'testing'
    ]

    text_lower = text.lower()
    skills = set()

    # Extract skills using pattern matching
    for pattern in skill_patterns:
        if pattern in text_lower:
            skills.add(pattern)

    # Process with spaCy
    doc = nlp(text_lower)

    # Extract skills from noun phrases
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        # Check if chunk contains technical terms
        if any(pattern in chunk_text for pattern in skill_patterns):
            if 2 <= len(chunk_text.split()) <= 4:  # Reasonable length for skill names
                skills.add(chunk_text)

    # Extract potential skills from named entities
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:
            ent_text = ent.text.lower().strip()
            if any(pattern in ent_text for pattern in skill_patterns):
                if len(ent_text.split()) <= 3:  # Keep entity names concise
                    skills.add(ent_text)

    print(f"Extracted {len(skills)} unique skills from text")
    return list(skills)

def extract_experience(text):
    """Extract work experience details."""
    if not text:
        return []

    experiences = []
    experience_patterns = [
        r'(\d{4})\s*-\s*(\d{4}|present)',
        r'(\w+ \d{4})\s*-\s*(\w+ \d{4}|present)',
        r'(\d{4})\s*to\s*(\d{4}|present)',
        r'(\w+ \d{4})\s*to\s*(\w+ \d{4}|present)'
    ]

    for pattern in experience_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            experiences.append(match.group())

    return experiences

def extract_education(text):
    """Extract education details."""
    if not text:
        return []

    education = []
    education_patterns = [
        r'(bachelor|master|phd|b\.?s\.?|m\.?s\.?|ph\.?d\.?)',
        r'(university|college|institute) of',
        r'(degree|diploma) in'
    ]

    for pattern in education_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            education.append(match.text.strip())

    return education