import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()


class Chain:
    def __init__(self):
        # Check if GROQ API key is available
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set your GROQ API key in the Streamlit secrets or environment variables."
            )
        
        try:
            self.llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model_name="llama3-70b-8192"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGroq: {str(e)}")
            
        # Your portfolio link
        self.portfolio_link = "https://anup2003d.github.io/portfolio-site/"

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
{page_data}

### INSTRUCTION:
Extract job posting information from the scraped career page text. Focus on identifying:
- Specific role requirements and responsibilities
- Required technical skills and tools
- Experience level and qualifications
- Company pain points or challenges mentioned
- Preferred qualifications or nice-to-haves
- Company culture indicators
- Company name (if mentioned)

Return the information in JSON format with the following keys: `role`, `experience`, `skills`, `description`, `requirements`, `company_challenges`, `preferred_skills`, `company_name`.
If any field is not found, use "Not specified" as the value.

### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def _detect_role_level(self, job_data):
        """Detect the seniority level of the role to adjust positioning"""
        if isinstance(job_data, dict):
            role = job_data.get('role', '').lower()
            description = job_data.get('description', '').lower()
            experience = job_data.get('experience', '').lower()
        else:
            role = str(job_data).lower()
            description = role
            experience = role

        # Check for role level indicators
        if any(word in role or word in description for word in ['intern', 'internship', 'trainee']):
            return 'internship'
        elif any(word in role or word in description for word in
                 ['junior', 'entry', 'associate', '0-2 years', 'fresher']):
            return 'junior'
        elif any(word in role or word in description for word in
                 ['senior', 'lead', 'principal', '5+ years', 'experienced']):
            return 'senior'
        else:
            return 'mid'  # Default to mid-level

    def write_mail(self, job, links=None):
        # Use provided links or default to your portfolio
        if links is None:
            links = [self.portfolio_link]
        elif isinstance(links, str):
            links = [links]

        # Extract company name from job data or use default
        company_name = job.get('company_name', 'the company') if isinstance(job, dict) else 'the company'

        # Detect role level for appropriate positioning
        role_level = self._detect_role_level(job)

        # Format job description properly
        job_description = str(job) if not isinstance(job, dict) else f"""
Role: {job.get('role', 'Not specified')}
Experience: {job.get('experience', 'Not specified')}
Skills: {job.get('skills', 'Not specified')}
Description: {job.get('description', 'Not specified')}
Requirements: {job.get('requirements', 'Not specified')}
Company Challenges: {job.get('company_challenges', 'Not specified')}
Preferred Skills: {job.get('preferred_skills', 'Not specified')}
Role Level Detected: {role_level}
        """

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
{job_description}

### COMPANY CONTEXT:
Company Name: {company_name}

### YOUR PROFILE (ANUP):
Professional Background:
- Data Analyst with 3+ years of experience in business intelligence and analytics
- Strong expertise in Python, SQL, Power BI, Tableau, and advanced Excel
- Experience with machine learning algorithms, predictive modeling, and statistical analysis
- Proven track record of transforming raw data into actionable business insights
- Background in data visualization, dashboard creation, and automated reporting systems
- Experience working with cross-functional teams to drive data-driven decision making

Key Achievements:
- Developed predictive models that improved forecasting accuracy by 30%
- Created automated dashboards that reduced manual reporting time by 60%
- Led data analysis projects that identified cost-saving opportunities worth $200K+
- Built ETL pipelines processing 1M+ records daily with 99.8% accuracy
- Mentored junior analysts and contributed to team knowledge sharing initiatives

Education & Certifications:
- Bachelor's degree in relevant field (Computer Science/Statistics/Engineering)
- Certified in advanced analytics tools and methodologies
- Continuous learner staying updated with latest data science trends

### PORTFOLIO LINKS:
{link_list}

### INSTRUCTION:
You are Anup, a Data Analyst seeking new opportunities. Write a compelling job application email that positions you as an ideal candidate for the specific role mentioned in the job description.

**CRITICAL: This is a JOB APPLICATION EMAIL, not a sales pitch. You are applying TO them, not selling services.**

**Structure Requirements:**
1. **Subject Line**: Professional application subject mentioning the specific role title
2. **Opening**: Express genuine interest in the specific role and company
3. **Skills Match**: Directly connect your technical skills to their requirements
4. **Relevant Experience**: Share 2-3 specific examples that demonstrate your ability to excel in this role
5. **Cultural Fit**: Show understanding of their company values and how you align
6. **Portfolio Reference**: Naturally mention your portfolio as evidence of your capabilities
7. **Professional Close**: Express enthusiasm and request for interview/discussion

**Tone and Style:**
- Professional and respectful (you're the applicant)
- Enthusiastic but not desperate
- Confident in your abilities without being arrogant
- Specific and tailored to their exact requirements
- Humble and eager to contribute to their team
- Concise (under 250 words)

**Key Elements to Include:**
- Use "I am writing to apply for..." or similar application language
- Match YOUR skills to THEIR specific requirements mentioned in job description
- Show genuine research about their company and role
- Quantify your achievements with specific metrics
- Demonstrate how you can solve their problems or add value to their team
- Express enthusiasm about joining THEIR organization
- Request an interview or call to discuss further

**Avoid:**
- Sounding like you're selling services to them
- Mentioning AtliQ as your current company (focus on your individual profile)
- Generic application templates
- Overly casual or overly formal language
- Making assumptions about their needs beyond what's in the job description
- Using buzzwords without substance

**Important Context Adjustments:**
- If the role is "Senior" level, position yourself as experienced and ready for senior responsibilities
- If the role is "Junior/Entry-level," show eagerness to learn and grow
- If it's an "Internship," emphasize learning goals and how you can contribute while growing
- Match the seniority level in your language and expectations

### OUTPUT FORMAT:
Subject: Application for [Specific Role Title] - [Your Name]

Dear Hiring Manager / Dear [Company Name] Team,

[Email body - professional job application tone]

Thank you for considering my application. I look forward to hearing from you.

Best regards,
Anup
Email: anup.analyst@gmail.com
Portfolio: {portfolio_link}
LinkedIn: linkedin.com/in/anup-data-analyst

############# EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": job_description,
            "company_name": company_name,
            "link_list": "\n".join([f"- {link}" for link in links]),
            "portfolio_link": self.portfolio_link
        })
        return res.content

    def generate_cold_email(self, job_data, custom_links=None):
        """
        Convenience method to generate a cold email from job data

        Args:
            job_data: Dictionary containing job information or raw job description string
            custom_links: Optional list of specific portfolio links to include

        Returns:
            Generated cold email as string
        """
        return self.write_mail(job_data, custom_links)


if __name__ == "__main__":
    # Test the code
    print("Testing GROQ API Key:", "✓ Found" if os.getenv("GROQ_API_KEY") else "✗ Not found")

    # Example usage:
    chain = Chain()

    # Sample job data for testing
    sample_job = {
        "role": "Data Analyst",
        "experience": "2-4 years",
        "skills": "Python, SQL, Tableau, Power BI",
        "description": "Looking for a data analyst to help with business intelligence and reporting",
        "requirements": "Strong SQL skills, experience with visualization tools",
        "company_challenges": "Need to improve data-driven decision making",
        "preferred_skills": "Machine learning, cloud platforms",
        "company_name": "TechCorp"
    }

    # Generate email
    try:
        email = chain.generate_cold_email(sample_job)
        print("\n" + "=" * 50)
        print("GENERATED EMAIL:")
        print("=" * 50)
        print(email)
    except Exception as e:
        print(f"Error generating email: {e}")